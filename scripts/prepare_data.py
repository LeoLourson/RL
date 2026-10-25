"""KuaiRec -> data/processed/kuairec: события, разбиение по времени, метаданные."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import N_DUR_BUCKETS, add_features, duration_bucket, load_raw, time_split

p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/raw/kuairec/KuaiRec 2.0/data")
p.add_argument("--out", default="data/processed/kuairec")
args = p.parse_args()
raw, out = Path(args.raw), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

big = load_raw(raw, "big_matrix")
split = time_split(big)
big = add_features(big, split)
small = load_raw(raw, "small_matrix")

users = big["user_id"].unique().sort()
items = big["video_id"].unique().sort()
umap = pl.DataFrame({"user_id": users, "uid": np.arange(len(users))})
imap = pl.DataFrame({"video_id": items, "iid": np.arange(len(items))})
big = big.join(umap, on="user_id").join(imap, on="video_id").sort(["uid", "timestamp"])
big = big.with_columns(
    split=pl.when(pl.col("timestamp") < split.t1).then(pl.lit("train"))
    .when(pl.col("timestamp") < split.t2).then(pl.lit("val")).otherwise(pl.lit("test"))
)

# популярность — только по train; длительность — статический атрибут видео, берём по всем данным
train = big.filter(pl.col("split") == "train")
dur = big.group_by("iid").agg(pl.col("video_duration").median()).sort("iid")
dur_ms = np.zeros(len(items)); dur_ms[dur["iid"].to_numpy()] = dur["video_duration"].fill_null(0).to_numpy()
dur_edges = np.quantile(dur_ms[dur_ms > 0], np.linspace(0, 1, N_DUR_BUCKETS + 1)[1:-1])
pop = np.zeros(len(items)); c = train.group_by("iid").len()
pop[c["iid"].to_numpy()] = c["len"].to_numpy()

big.select(
    "uid", "iid", "timestamp", "watch_ratio", "ratio", "play_duration", "video_duration",
    "cont_session", "return_24h", "observed", "split",
).write_parquet(out / "events.parquet")
np.save(out / "item_dur_bucket.npy", duration_bucket(dur_ms, dur_edges))
np.save(out / "item_dur_ms.npy", dur_ms)
np.save(out / "item_pop.npy", pop)

# оракул: полностью наблюдаемая small_matrix в тех же индексах
small = small.join(umap, on="user_id").join(imap, on="video_id")
small.select("uid", "iid", "timestamp", "watch_ratio").write_parquet(out / "small_oracle.parquet")

overlap = small.join(big.select("uid", "iid"), on=["uid", "iid"], how="semi").height
meta = {
    "n_users": len(users), "n_items": len(items), "t1": split.t1, "t2": split.t2,
    "dur_edges": dur_edges.tolist(),
    "split_counts": dict(big.group_by("split").len().iter_rows()),
    "small_users": small["uid"].n_unique(), "small_items": small["iid"].n_unique(),
    "small_rows": small.height, "small_pairs_in_big": overlap,
}
(out / "meta.json").write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
