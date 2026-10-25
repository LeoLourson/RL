"""Единый протокол оценки для SFT и RL-моделей."""
from __future__ import annotations

import numpy as np
import polars as pl
import torch

from .data import PAD, UserSeqs, eval_contexts, ratio_bucket
from .metrics import behavior_metrics, ranking_metrics


def left_ctx(seqs: UserSeqs, rows, T: int):
    """Контексты [B, T] с паддингом справа + отклики для reward model."""
    items = np.zeros((len(rows), T), dtype=np.int64)
    ratio = np.zeros((len(rows), T), dtype=np.float32)
    for b, (u, s, a, _) in enumerate(rows):
        L = a - s
        items[b, :L] = seqs.items[u][s:a]
        ratio[b, :L] = seqs.ratio[u][s:a]
    return torch.from_numpy(items), torch.from_numpy(ratio)


@torch.no_grad()
def policy_topk(policy, items: torch.Tensor, K: int = 20, bs: int = 512, allowed: torch.Tensor | None = None,
                exclude: list[set] | None = None):
    out = []
    for i in range(0, len(items), bs):
        lg = policy.last_logits(items[i:i + bs])
        if allowed is not None:
            lg = lg.masked_fill(~allowed, -1e9)
        if exclude is not None:
            for j, h in enumerate(exclude[i:i + bs]):
                lg[j, list(h)] = -1e9
        out.append(lg.topk(K, -1).indices)
    return torch.cat(out).numpy()


@torch.no_grad()
def rm_scores(rm, items, ratio, cands, bs: int = 512):
    """Предсказания RM для кандидатов cands [B, K] после контекста."""
    res = {k: [] for k in ("ratio", "cont", "ret")}
    for i in range(0, len(items), bs):
        it, r = items[i:i + bs], ratio[i:i + bs]
        h = rm.encode(it, torch.from_numpy(ratio_bucket(r.numpy())))
        last = (it != PAD).sum(1) - 1
        hl = h[torch.arange(len(it)), last]
        c = torch.as_tensor(cands[i:i + bs])
        p = rm.to_preds(rm.score(hl[:, None].expand(-1, c.size(1), -1), c))
        for k in res:
            res[k].append(p[k])
    return {k: torch.cat(v).numpy() for k, v in res.items()}


class Evaluator:
    def __init__(self, seqs: UserSeqs, data_dir, T: int, stage: str = "val"):
        self.seqs, self.T = seqs, T
        sp = seqs.split
        t_from, t_to = (sp.t1, sp.t2) if stage == "val" else (sp.t2, float("inf"))
        self.rows = eval_contexts(seqs, t_from, t_to, T)
        self.items, self.ratio = left_ctx(seqs, self.rows, T)
        self.gts = [set(seqs.items[u][a:b].tolist()) for u, _, a, b in self.rows]
        self.gts_pos = [set(seqs.items[u][a:b][seqs.ratio[u][a:b] >= 2.0].tolist()) for u, _, a, b in self.rows]
        self.hist = [set(seqs.items[u][:a].tolist()) for u, _, a, _ in self.rows]
        self.gts_new = [g - h for g, h in zip(self.gts, self.hist)]
        self.dur_ms = np.load(f"{data_dir}/item_dur_ms.npy")
        self._oracle(data_dir)

    def _oracle(self, data_dir):
        """small_matrix: истинная доля досмотра для 1411 пользователей × 3327 объектов."""
        sm = pl.read_parquet(f"{data_dir}/small_oracle.parquet")
        self.small_items = np.sort(sm["iid"].unique().to_numpy()) + 1
        uids = {u: i for i, (u, *_) in enumerate(self.rows)}
        sm = sm.filter(pl.col("uid").is_in(list(uids)))
        self.o_rows = np.array([uids[u] for u in sm["uid"].unique().sort().to_list()])
        self.o_users = sm["uid"].unique().sort().to_numpy()
        self.oracle = {
            (u, i + 1): min(r, 3.0)
            for u, i, r in sm.select("uid", "iid", "watch_ratio").iter_rows()
        }

    def run(self, policy, rm=None, K: int = 20) -> dict:
        policy.eval()
        top = policy_topk(policy, self.items, K)
        res = ranking_metrics(top, self.gts)
        pos = [i for i, g in enumerate(self.gts_pos) if g]
        res |= {f"pos_{k}": v for k, v in ranking_metrics(top[pos], [self.gts_pos[i] for i in pos]).items()
                if k.endswith("@10")}
        # протокол без повторов: история исключена и из топа, и из ground truth
        top_new = policy_topk(policy, self.items, K, exclude=self.hist)
        nz = [i for i, g in enumerate(self.gts_new) if g]
        res |= {f"new_{k}": v for k, v in ranking_metrics(top_new[nz], [self.gts_new[i] for i in nz]).items()
                if k.endswith("@10")}
        res |= behavior_metrics(top, self.seqs.n_items, self.seqs.item_pop, self.dur_ms, self.hist)
        if rm is not None:
            rm.eval()
            s = rm_scores(rm, self.items, self.ratio, top[:, :10])
            res |= {f"RM_{k}@10": float(v.mean()) for k, v in s.items()}
        # оракул: каталог ограничен объектами small_matrix
        allowed = torch.zeros(self.seqs.n_items + 1, dtype=torch.bool)
        allowed[torch.from_numpy(self.small_items)] = True
        ot = policy_topk(policy, self.items[self.o_rows], 10, allowed=allowed)
        vals = [self.oracle.get((u, i)) for u, row in zip(self.o_users, ot) for i in row]
        vals = [v for v in vals if v is not None]
        res["Oracle_ratio@10"] = float(np.mean(vals))
        res["Oracle_coverage"] = len(vals) / ot.size
        if rm is not None:
            s = rm_scores(rm, self.items[self.o_rows], self.ratio[self.o_rows], ot)
            res["RM_ratio_on_oracle@10"] = float(s["ratio"].mean())
        return res
