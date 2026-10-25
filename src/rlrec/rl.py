"""RLOO-дообучение политики по reward model с KL-штрафом к SFT-модели."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import PAD, RATIO_CLIP, UserSeqs, ratio_bucket


def sample_contexts(seqs: UserSeqs, B: int, T: int, rng: np.random.Generator, min_len: int = 10):
    """Случайные отсечки внутри train-периода: контекст — до T событий перед отсечкой."""
    items = np.zeros((B, T), dtype=np.int64)
    ratio = np.zeros((B, T), dtype=np.float32)
    hist = []
    b = 0
    while b < B:
        u = int(rng.integers(len(seqs.items)))
        n = seqs.n_before(u, seqs.split.t1)
        if n <= min_len:
            continue
        c = int(rng.integers(min_len, n + 1))
        s = max(0, c - T)
        items[b, : c - s] = seqs.items[u][s:c]
        ratio[b, : c - s] = seqs.ratio[u][s:c]
        hist.append(seqs.items[u][:c])
        b += 1
    return torch.from_numpy(items), torch.from_numpy(ratio), hist


def append(items, ratio, a, r):
    """Дописать объект a с откликом r в конец контекста; при переполнении сдвинуть влево."""
    items, ratio = items.clone(), ratio.clone()  # старые тензоры нужны графу градиента
    L = (items != PAD).sum(1)
    full = L >= items.size(1)
    if full.any():
        items[full] = torch.roll(items[full], -1, 1); ratio[full] = torch.roll(ratio[full], -1, 1)
        L = torch.where(full, L - 1, L)
    idx = torch.arange(len(items))
    items[idx, L] = a; ratio[idx, L] = r
    return items, ratio


class RewardFn:
    """Варианты вознаграждения поверх предсказаний RM.
    ratio  — доля досмотра;
    time   — ожидаемое время просмотра (ratio × длительность, нормировано на среднюю длительность);
    cont   — доля досмотра + α·P(продолжения сессии) + α·P(возврата);
    Общие добавки: λ_rel·log π_ref(a) (якорь релевантности), штраф δ за повтор просмотренного."""

    def __init__(self, rm, kind: str, item_dur_ms: np.ndarray, alpha=1.0, lam_rel=0.0, rep_pen=0.0):
        self.rm, self.kind, self.alpha, self.lam_rel, self.rep_pen = rm, kind, alpha, lam_rel, rep_pen
        d = torch.as_tensor(item_dur_ms, dtype=torch.float32)
        self.dur = torch.cat([torch.zeros(1), d / d[d > 0].mean()])

    @torch.no_grad()
    def __call__(self, items, ratio, acts, ref_logp=None, seen=None):
        """items/ratio: [B, T], acts: [B, K] -> (reward [B, K], предсказания RM)."""
        h = self.rm.encode(items, torch.from_numpy(ratio_bucket(ratio.numpy())))
        last = (items != PAD).sum(1) - 1
        hl = h[torch.arange(len(items)), last]
        p = self.rm.to_preds(self.rm.score(hl[:, None].expand(-1, acts.size(1), -1), acts))
        if self.kind == "ratio":
            r = p["ratio"]
        elif self.kind == "time":
            r = p["ratio"] * self.dur[acts]
        elif self.kind == "cont":
            r = p["ratio"] + self.alpha * (p["cont"] + p["ret"])
        else:
            raise ValueError(self.kind)
        if self.lam_rel and ref_logp is not None:
            r = r + self.lam_rel * ref_logp
        if self.rep_pen and seen is not None:
            r = r - self.rep_pen * seen.float()
        return r, p


def rloo_step(policy, ref, reward_fn, items, ratio, hist, K: int, beta: float, H: int = 1,
              gamma: float = 0.9, temp: float = 1.0):
    """Один шаг RLOO. При H>1 политика продолжает траекторию, добавляя свой объект и
    предсказанный RM отклик в контекст; возврат — дисконтированная сумма наград."""
    B = len(items)
    it = items.repeat_interleave(K, 0).clone(); rt = ratio.repeat_interleave(K, 0).clone()
    seen_sets = [set(h.tolist()) for h in hist for _ in range(K)]
    logps, rewards, kls, info = [], [], [], []
    for step in range(H):
        # на первом шаге K копий контекста одинаковы: считаем политику на B, а не на B*K
        src = items if step == 0 else it
        logits = policy.last_logits(src) / temp
        logp = F.log_softmax(logits, -1)
        with torch.no_grad():
            ref_logp = F.log_softmax(ref.last_logits(src), -1)
        if step == 0:
            logp, ref_logp = logp.repeat_interleave(K, 0), ref_logp.repeat_interleave(K, 0)
        with torch.no_grad():
            a = torch.multinomial(logp.exp(), 1).squeeze(1)
        p = logp.exp()
        kls.append((p * (logp - ref_logp)).sum(-1))  # точный KL(π || π_ref) по каталогу
        logps.append(logp.gather(1, a[:, None]).squeeze(1))
        seen = torch.tensor([int(x) in s for x, s in zip(a.tolist(), seen_sets)])
        ra = ref_logp.gather(1, a[:, None]).squeeze(1)
        if step == 0:  # RM тоже кодирует только B уникальных контекстов
            r, pr = reward_fn(items, ratio, a.view(B, K), ra.view(B, K), seen.view(B, K))
        else:
            r, pr = reward_fn(it, rt, a[:, None], ra[:, None], seen[:, None])
        r, pr_ratio = r.reshape(-1), pr["ratio"].reshape(-1)
        rewards.append(r)
        info.append({"a": a, "seen": seen, "ratio_pred": pr_ratio})
        if step + 1 < H:
            # отклик сэмплируем из Beta со средним E[ratio]: подстановка среднего дала бы
            # RM входы, которых не было в обучении (всегда средние корзины)
            m = (pr_ratio / RATIO_CLIP).clamp(0.02, 0.98)
            r_s = torch.distributions.Beta(2 * m, 2 * (1 - m)).sample() * RATIO_CLIP
            it, rt = append(it, rt, a, r_s)
            for s, x in zip(seen_sets, a.tolist()):
                s.add(x)
    R = torch.stack(rewards, 1)  # [B*K, H]
    G = torch.zeros_like(R); acc = torch.zeros(B * K)
    for t in reversed(range(H)):
        acc = R[:, t] + gamma * acc; G[:, t] = acc
    G = G.view(B, K, H)
    adv = (G - (G.sum(1, keepdim=True) - G) / (K - 1)).view(B * K, H)  # leave-one-out baseline
    adv = adv / (adv.std() + 1e-6)
    pg = -(adv.detach() * torch.stack(logps, 1)).mean()
    kl = torch.stack(kls, 1).mean()
    stats = {
        "reward": R[:, 0].mean().item(), "return": G[:, :, 0].mean().item(), "kl": kl.item(),
        "repeat": info[0]["seen"].float().mean().item(), "ratio_pred": info[0]["ratio_pred"].mean().item(),
        "uniq_frac": len(set(info[0]["a"].tolist())) / (B * K),
    }
    return pg + beta * kl, stats
