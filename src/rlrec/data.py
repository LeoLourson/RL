"""Загрузка KuaiRec, глобальное разбиение по времени и батчи для обучения."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch

PAD = 0  # токен паддинга; id объекта i кодируется как i + 1
SESSION_GAP_S = 30 * 60
RATIO_CLIP = 3.0
N_RATIO_BUCKETS = 12
N_DUR_BUCKETS = 8


def load_raw(raw_dir: Path, name: str) -> pl.DataFrame:
    df = pl.read_csv(raw_dir / f"{name}.csv")
    return (
        df.filter(pl.col("timestamp").is_not_null() & (pl.col("timestamp") > 0))
        .with_columns(pl.col("timestamp").cast(pl.Float64))
        .sort(["user_id", "timestamp"])
    )


def add_features(df: pl.DataFrame, split: "Split") -> pl.DataFrame:
    """Сессии, сигналы продолжения и возврата для долгосрочной награды.
    Разрыв до следующего события считается внутри сплита, чтобы метки train не видели val.
    Последнее событие пользователя в сплите цензурировано (observed = 0)."""
    part = (pl.col("timestamp") >= split.t1).cast(pl.Int8) + (pl.col("timestamp") >= split.t2).cast(pl.Int8)
    df = df.with_columns(part=part).with_columns(
        gap_next=pl.col("timestamp").shift(-1).over(["user_id", "part"]) - pl.col("timestamp"),
    )
    return df.with_columns(
        observed=pl.col("gap_next").is_not_null().cast(pl.Int8),
        # пользователь продолжил смотреть в той же сессии
        cont_session=(pl.col("gap_next") < SESSION_GAP_S).fill_null(False).cast(pl.Int8),
        # событие последнее в сессии и пользователь вернулся в течение суток
        return_24h=(
            (pl.col("gap_next") >= SESSION_GAP_S) & (pl.col("gap_next") < 24 * 3600)
        ).fill_null(False).cast(pl.Int8),
        ratio=pl.col("watch_ratio").clip(0.0, RATIO_CLIP).cast(pl.Float32),
    ).drop("gap_next", "part")


def ratio_bucket(r: np.ndarray) -> np.ndarray:
    edges = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0])
    return np.digitize(r, edges).astype(np.int64)  # 0..11


def duration_bucket(ms: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(ms, edges).astype(np.int64)  # 0..N_DUR_BUCKETS-1


@dataclass
class Split:
    t1: float  # начало val
    t2: float  # начало test


def time_split(df: pl.DataFrame, val_q: float = 0.8, test_q: float = 0.9) -> Split:
    ts = df["timestamp"]
    return Split(float(ts.quantile(val_q)), float(ts.quantile(test_q)))


class UserSeqs:
    """Хронологические последовательности пользователей в виде numpy-массивов."""

    def __init__(self, path: Path):
        df = pl.read_parquet(path / "events.parquet")
        meta = json.loads((path / "meta.json").read_text())
        self.n_items = meta["n_items"]
        self.split = Split(meta["t1"], meta["t2"])
        self.dur_edges = np.array(meta["dur_edges"])
        self.item_dur = np.load(path / "item_dur_bucket.npy")
        self.item_pop = np.load(path / "item_pop.npy")
        g = df.group_by("uid", maintain_order=True).agg(
            "iid", "timestamp", "ratio", "cont_session", "return_24h", "observed"
        )
        self.uids = g["uid"].to_numpy()
        self.items = [np.asarray(x, dtype=np.int64) + 1 for x in g["iid"].to_list()]
        self.ts = [np.asarray(x) for x in g["timestamp"].to_list()]
        self.ratio = [np.asarray(x, dtype=np.float32) for x in g["ratio"].to_list()]
        self.cont = [np.asarray(x, dtype=np.float32) for x in g["cont_session"].to_list()]
        self.ret = [np.asarray(x, dtype=np.float32) for x in g["return_24h"].to_list()]
        self.obs = [np.asarray(x, dtype=np.float32) for x in g["observed"].to_list()]

    def n_before(self, u: int, t: float) -> int:
        return int(np.searchsorted(self.ts[u], t, side="left"))


def sample_train_windows(seqs: UserSeqs, T: int, rng: np.random.Generator):
    """Нарезка train-части каждого пользователя на окна длины T+1 со случайным сдвигом."""
    out = []
    for u in range(len(seqs.items)):
        n = seqs.n_before(u, seqs.split.t1)
        if n < 3:
            continue
        start = int(rng.integers(0, T)) % max(1, n - 1)
        if start > 0:
            out.append((u, 0, start + 1))
        for s in range(start, n - 1, T):
            out.append((u, s, min(s + T + 1, n)))
    rng.shuffle(out)
    return out


def collate(seqs: UserSeqs, windows, T: int, with_feedback: bool = False):
    B = len(windows)
    items = np.zeros((B, T + 1), dtype=np.int64)
    ratio = np.zeros((B, T + 1), dtype=np.float32)
    cont = np.zeros((B, T + 1), dtype=np.float32)
    ret = np.zeros((B, T + 1), dtype=np.float32)
    obs = np.zeros((B, T + 1), dtype=np.float32)
    for b, (u, s, e) in enumerate(windows):
        L = e - s
        items[b, :L] = seqs.items[u][s:e]  # паддинг справа
        if with_feedback:
            ratio[b, :L] = seqs.ratio[u][s:e]
            cont[b, :L] = seqs.cont[u][s:e]
            ret[b, :L] = seqs.ret[u][s:e]
            obs[b, :L] = seqs.obs[u][s:e]
    batch = {"items": torch.from_numpy(items)}
    if with_feedback:
        batch |= {
            "ratio": torch.from_numpy(ratio),
            "rbucket": torch.from_numpy(ratio_bucket(ratio)),
            "cont": torch.from_numpy(cont),
            "ret": torch.from_numpy(ret),
            "obs": torch.from_numpy(obs),
        }
    return batch


def eval_contexts(seqs: UserSeqs, t_from: float, t_to: float, T: int, min_gt: int = 1):
    """Для каждого пользователя: контекст — последние T событий до t_from,
    ground truth — объекты из интервала [t_from, t_to)."""
    rows = []
    for u in range(len(seqs.items)):
        a, b = seqs.n_before(u, t_from), seqs.n_before(u, t_to)
        if a < 1 or b - a < min_gt:
            continue
        rows.append((u, max(0, a - T), a, b))
    return rows
