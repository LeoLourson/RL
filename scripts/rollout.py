"""Генерация H шагов вперёд: сгенерированный объект добавляется в контекст.
На каждом шаге сравниваем с «реальной» веткой, где в контекст добавляется
фактическое следующее событие пользователя (teacher forcing)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import PAD, RATIO_CLIP, UserSeqs
from rlrec.evaluate import Evaluator, rm_scores
from rlrec.model import Policy, RewardModel
from rlrec.rl import append

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/kuairec")
p.add_argument("--rm", default="artifacts/rm/model.pt")
p.add_argument("--policies", nargs="+", required=True, help="name=path/model.pt")
p.add_argument("--H", type=int, default=5)
p.add_argument("--feedback", default="sample", choices=["sample", "mean"])
p.add_argument("--no_repeat", action="store_true", help="запретить повтор уже сгенерированных объектов")
p.add_argument("--out", default="artifacts/rollout.json")
p.add_argument("--threads", type=int, default=16)
args = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(args.threads)

seqs = UserSeqs(Path(args.data))
ev = Evaluator(seqs, args.data, 100, "test")
rk = torch.load(args.rm)
rm = RewardModel(GPT2Config.from_dict(rk["cfg"]), torch.from_numpy(seqs.item_dur))
rm.load_state_dict(rk["state"]); rm.eval()
dur = np.load(f"{args.data}/item_dur_ms.npy") / 1000

# для оракула каталог ограничен объектами small_matrix
allowed = torch.zeros(seqs.n_items + 1, dtype=torch.bool); allowed[torch.from_numpy(ev.small_items)] = True
orow = {r: u for r, u in zip(ev.o_rows, ev.o_users)}


@torch.no_grad()
def run(policy, rows_idx, restrict):
    it = torch.cat([ev.items[rows_idx], torch.zeros(len(rows_idx), args.H, dtype=torch.long)], 1)
    rt = torch.cat([ev.ratio[rows_idx], torch.zeros(len(rows_idx), args.H)], 1)
    # «реальная» ветка: дописываем фактические следующие события
    it_r, rt_r = it.clone(), rt.clone()
    futures = [seqs.items[ev.rows[i][0]][ev.rows[i][2]:ev.rows[i][2] + args.H] for i in rows_idx]
    fut_r = [seqs.ratio[ev.rows[i][0]][ev.rows[i][2]:ev.rows[i][2] + args.H] for i in rows_idx]
    gen_sets = [set() for _ in rows_idx]
    per_step = []
    for h in range(args.H):
        lg = policy.last_logits(it); lg_r = policy.last_logits(it_r)
        if restrict:
            lg = lg.masked_fill(~allowed, -1e9); lg_r = lg_r.masked_fill(~allowed, -1e9)
        if args.no_repeat:  # без запрета жадная генерация зацикливается на только что дописанном объекте
            for j, s in enumerate(gen_sets):
                if s:
                    lg[j, list(s)] = -1e9
        a = lg.argmax(-1)
        top_gen, top_real = lg.topk(10, -1).indices, lg_r.topk(10, -1).indices
        pr = rm_scores(rm, it, rt, a[:, None].numpy())["ratio"][:, 0]
        st = {
            "step": h + 1,
            "RM_ratio": float(pr.mean()),
            "repeat_hist": float(np.mean([int(x) in ev.hist[i] for x, i in zip(a.tolist(), rows_idx)])),
            "repeat_gen": float(np.mean([int(x) in s for x, s in zip(a.tolist(), gen_sets)])),
            "dur_s": float(dur[a.numpy() - 1].mean()),
            "pop": float(seqs.item_pop[a.numpy() - 1].mean()),
            # пересечение топ-10 сгенерированной и реальной веток: мера дрейфа контекста
            "overlap_top10_vs_real_ctx": float(np.mean([len(set(x.tolist()) & set(y.tolist())) / 10
                                                         for x, y in zip(top_gen, top_real)])),
            "uniq_items": int(len(set(a.tolist()))),
        }
        if not restrict:
            gt = [ev.gts[i] for i in rows_idx]
            st["Recall@10_gen_ctx"] = float(np.mean([len(set(x.tolist()) & g) / min(10, len(g)) for x, g in zip(top_gen, gt)]))
            st["Recall@10_real_ctx"] = float(np.mean([len(set(x.tolist()) & g) / min(10, len(g)) for x, g in zip(top_real, gt)]))
        else:
            tv = [ev.oracle.get((orow[i], int(x))) for x, i in zip(a.tolist(), rows_idx)]
            tv = [v for v in tv if v is not None]
            st["Oracle_ratio"] = float(np.mean(tv)); st["RM_minus_oracle"] = st["RM_ratio"] - st["Oracle_ratio"]
        per_step.append(st)
        # отклик на сгенерированный объект: предсказание RM (сэмпл или среднее)
        r = torch.from_numpy(pr)
        if args.feedback == "sample":
            m = (r / RATIO_CLIP).clamp(0.02, 0.98)
            r = torch.distributions.Beta(2 * m, 2 * (1 - m)).sample() * RATIO_CLIP
        it, rt = append(it, rt, a, r)
        for s, x in zip(gen_sets, a.tolist()):
            s.add(x)
        ok = torch.tensor([len(f) > h for f in futures])
        nxt = torch.tensor([int(f[h]) if len(f) > h else PAD for f in futures])
        nr = torch.tensor([float(f[h]) if len(f) > h else 0.0 for f in fut_r])
        it_n, rt_n = append(it_r, rt_r, nxt, nr)
        it_r[ok], rt_r[ok] = it_n[ok], rt_n[ok]
    return per_step


res = {}
for spec in args.policies:
    name, path = spec.split("=", 1)
    ck = torch.load(path)
    pol = Policy(GPT2Config.from_dict(ck["cfg"])); pol.load_state_dict(ck["state"]); pol.eval()
    res[name] = {
        "all_users": run(pol, list(range(len(ev.rows))), False),
        "oracle_users": run(pol, list(ev.o_rows), True),
    }
    print(name, json.dumps(res[name], indent=1), flush=True)
Path(args.out).write_text(json.dumps(res, indent=2))
