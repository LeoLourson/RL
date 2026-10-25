"""Графики: компромисс релевантность/награда (test) и траектории обучения (val)."""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = Path("reports/figures"); out.mkdir(parents=True, exist_ok=True)
runs = {"sft": json.load(open("artifacts/sft/test.json"))}
for f in sorted(glob.glob("artifacts/rl/*/test.json")):
    runs[Path(f).parent.name] = json.load(open(f))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, y in zip(axes, ["RM_ratio@10", "Oracle_ratio@10"]):
    for n, d in runs.items():
        ax.scatter(d["NDCG@10"], d[y], s=40, c="k" if n == "sft" else None)
        ax.annotate(n, (d["NDCG@10"], d[y]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("NDCG@10 (test)"); ax.set_ylabel(y + " (test)"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(out / "tradeoff_test.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for f in sorted(glob.glob("artifacts/rl/*/log.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    ev = [r for r in rows if "step" in r and "train" not in r]
    if len(ev) < 2:
        continue
    n = Path(f).parent.name
    axes[0].plot([r["NDCG@10"] for r in ev], [r["Oracle_ratio@10"] for r in ev], "o-", label=n)
    axes[1].plot([r["step"] for r in ev], [r["Uniq_top1"] for r in ev], "o-", label=n)
    axes[2].plot([r["step"] for r in ev], [r["MeanDur_s"] for r in ev], "o-", label=n)
axes[0].set_xlabel("NDCG@10 (val)"); axes[0].set_ylabel("Oracle_ratio@10 (val)")
axes[1].set_xlabel("шаг RL"); axes[1].set_ylabel("Uniq_top1 (val)")
axes[2].set_xlabel("шаг RL"); axes[2].set_ylabel("средняя длительность топ-10, с")
for a in axes:
    a.grid(alpha=.3)
axes[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(out / "training_val.png", dpi=130)

for name in ["rollout", "rollout_norep"]:
    p = Path(f"artifacts/{name}.json")
    if not p.exists():
        continue
    d = json.load(open(p))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for n, r in d.items():
        a, o = r["all_users"], r["oracle_users"]
        st = [s["step"] for s in a]
        axes[0].plot(st, [s["Recall@10_gen_ctx"] for s in a], "o-", label=n + " (ген. контекст)")
        axes[0].plot(st, [s["Recall@10_real_ctx"] for s in a], "x--", label=n + " (реальный)")
        axes[1].plot(st, [s["overlap_top10_vs_real_ctx"] for s in a], "o-", label=n)
        axes[2].plot(st, [s["Oracle_ratio"] for s in o], "o-", label=n + " оракул")
        axes[2].plot(st, [s["RM_ratio"] for s in o], "x--", label=n + " RM")
    axes[0].set_ylabel("Recall@10 (test)"); axes[1].set_ylabel("пересечение топ-10 с реальной веткой")
    axes[2].set_ylabel("ratio выбранного объекта")
    for a in axes:
        a.set_xlabel("шаг генерации"); a.grid(alpha=.3); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=130)
print("ok")
