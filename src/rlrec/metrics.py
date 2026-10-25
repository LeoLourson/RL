"""Метрики релевантности и поведения рекомендаций."""
from __future__ import annotations

import numpy as np


def ranking_metrics(topk: np.ndarray, gts: list[set], ks=(10, 20)) -> dict:
    """topk: [U, Kmax] id объектов. gts: множества релевантных объектов.
    Recall@K нормирован на min(K, |GT|), релевантность бинарная."""
    res = {}
    for K in ks:
        rec, ndcg, ap = [], [], []
        disc = 1.0 / np.log2(np.arange(2, K + 2))
        for row, gt in zip(topk[:, :K], gts):
            hit = np.fromiter((i in gt for i in row), bool, K)
            m = min(K, len(gt))
            rec.append(hit.sum() / m)
            ndcg.append((hit * disc).sum() / disc[:m].sum())
            prec = np.cumsum(hit) / np.arange(1, K + 1)
            ap.append((prec * hit).sum() / m)
        res[f"Recall@{K}"] = float(np.mean(rec))
        res[f"NDCG@{K}"] = float(np.mean(ndcg))
        res[f"MAP@{K}"] = float(np.mean(ap))
    return res


def gini(counts: np.ndarray) -> float:
    x = np.sort(counts.astype(float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def behavior_metrics(topk: np.ndarray, n_items: int, item_pop: np.ndarray,
                     item_dur_ms: np.ndarray, histories: list[set], K: int = 10) -> dict:
    """Coverage, Gini, популярность, длительность, повторы, совпадение топ-1 между пользователями."""
    t = topk[:, :K]
    counts = np.bincount(t.ravel(), minlength=n_items + 1)[1:]
    top1 = np.bincount(t[:, 0], minlength=n_items + 1)
    rep = np.mean([np.mean([i in h for i in row]) for row, h in zip(t, histories)])
    return {
        f"Coverage@{K}": float((counts > 0).mean()),
        f"Gini@{K}": gini(counts),
        "Top1_share_max": float(top1.max() / len(t)),
        "Uniq_top1": int((top1 > 0).sum()),
        "MeanPop": float(item_pop[t - 1].mean()),
        "MeanDur_s": float(item_dur_ms[t - 1].mean() / 1000),
        "RepeatRate": float(rep),
    }
