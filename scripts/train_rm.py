"""Reward model: доля досмотра, продолжение сессии и возврат по истории с откликами.
Оценка на test-периоде (реальные следующие события) и на оракуле small_matrix."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import PAD, UserSeqs, collate, sample_train_windows
from rlrec.evaluate import Evaluator, rm_scores
from rlrec.model import RewardModel, gpt2_config

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/kuairec")
p.add_argument("--out", default="artifacts/rm")
p.add_argument("--T", type=int, default=100)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=2)
p.add_argument("--threads", type=int, default=16)
p.add_argument("--eval_only", action="store_true")
args = p.parse_args()
torch.manual_seed(0); torch.set_num_threads(args.threads)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
seqs = UserSeqs(Path(args.data))
cfg = gpt2_config(seqs.n_items, args.d, args.layers, args.heads, args.T)
rm = RewardModel(cfg, torch.from_numpy(seqs.item_dur))


def eval_windows(t_from, t_to, T):
    """Окна длины T+1: вторая половина — цели из [t_from, t_to), первая — контекст."""
    h, wins = T // 2, []
    for u in range(len(seqs.items)):
        a, b = seqs.n_before(u, t_from), seqs.n_before(u, t_to)
        for q in range(max(a, 1), b, h):
            s = max(0, q - h)
            wins.append((u, s, min(s + T + 1, b), q))
    return wins


@torch.no_grad()
def eval_events(t_from, t_to, max_users=None):
    rm.eval()
    wins = eval_windows(t_from, t_to, args.T)
    pred = {k: [] for k in ("ratio", "cont", "ret")}; tgt = {k: [] for k in pred}; dur = []
    for i in range(0, len(wins), 256):
        w = wins[i:i + 256]
        b = collate(seqs, [x[:3] for x in w], args.T, True)
        it = b["items"]
        hs = rm.encode(it[:, :-1], b["rbucket"][:, :-1])
        o = rm.to_preds(rm.score(hs, it[:, 1:]))
        m = (it[:, 1:] != PAD) & (it[:, :-1] != PAD)
        h = args.T // 2
        for j, (u, s, e, q) in enumerate(w):
            m[j, : q - s - 1] = False  # цели раньше t_from не оцениваем
            m[j, q + h - s - 1:] = False  # каждая цель ровно в одном окне: [q, q+h)
        mo = m & (b["obs"][:, 1:] > 0)
        pred["ratio"].append(o["ratio"][m].numpy()); tgt["ratio"].append(b["ratio"][:, 1:][m].numpy())
        for k in ("cont", "ret"):
            pred[k].append(o[k][mo].numpy()); tgt[k].append(b[k][:, 1:][mo].numpy())
        dur.append(rm.item_dur[it[:, 1:]][m].numpy())
    pred = {k: np.concatenate(v) for k, v in pred.items()}; tgt = {k: np.concatenate(v) for k, v in tgt.items()}
    dur = np.concatenate(dur)
    y, yh = tgt["ratio"], pred["ratio"]
    res = {
        "n": int(len(y)),
        "ratio_RMSE": float(np.sqrt(np.mean((y - yh) ** 2))),
        "ratio_RMSE_const": float(np.std(y)),
        "ratio_MAE": float(np.mean(np.abs(y - yh))),
        "ratio_Pearson": float(pearsonr(y, yh)[0]),
        "ratio_Spearman": float(spearmanr(y, yh)[0]),
        "AUC_ratio>=1": float(roc_auc_score(y >= 1, yh)),
        "AUC_ratio>=2": float(roc_auc_score(y >= 2, yh)),
        "AUC_cont": float(roc_auc_score(tgt["cont"], pred["cont"])),
        "AUC_ret": float(roc_auc_score(tgt["ret"], pred["ret"])),
        "mean_ratio_true": float(y.mean()), "mean_ratio_pred": float(yh.mean()),
    }
    # смещение по длительности: средняя ошибка в каждой корзине
    res["bias_by_dur"] = {int(d): float((yh - y)[dur == d].mean()) for d in np.unique(dur)}
    res["true_by_dur"] = {int(d): float(y[dur == d].mean()) for d in np.unique(dur)}
    return res


@torch.no_grad()
def eval_oracle():
    """Для пользователей small_matrix: контекст до t2, кандидаты — все объекты small_matrix.
    Сравниваем предсказание RM с истинной долей досмотра, глобально и внутри пользователя."""
    rm.eval()
    ev = Evaluator(seqs, args.data, args.T, "test")
    sm = pl.read_parquet(f"{args.data}/small_oracle.parquet").partition_by("uid", as_dict=True)
    ys, yhs, sp = [], [], []
    for r, u in zip(ev.o_rows, ev.o_users):
        su = sm[(u,)]
        cand = su["iid"].to_numpy() + 1
        y = np.minimum(su["watch_ratio"].to_numpy(), 3.0)
        yh = rm_scores(rm, ev.items[r:r + 1], ev.ratio[r:r + 1], cand[None])["ratio"][0]
        ys.append(y); yhs.append(yh); sp.append(spearmanr(y, yh)[0])
    y, yh = np.concatenate(ys), np.concatenate(yhs)
    # насколько верно RM выбирает топ-10 для пользователя
    top_true = [float(np.mean(a[np.argsort(-b)[:10]])) for a, b in zip(ys, yhs)]
    best = [float(np.mean(np.sort(a)[-10:])) for a in ys]
    return {
        "n_users": len(ys), "n": int(len(y)),
        "RMSE": float(np.sqrt(np.mean((y - yh) ** 2))), "RMSE_const": float(np.std(y)),
        "Pearson": float(pearsonr(y, yh)[0]), "Spearman": float(spearmanr(y, yh)[0]),
        "Spearman_within_user_mean": float(np.nanmean(sp)),
        "AUC_ratio>=2": float(roc_auc_score(y >= 2, yh)),
        "true_ratio_of_RM_top10": float(np.mean(top_true)),
        "true_ratio_random": float(y.mean()), "true_ratio_best10": float(np.mean(best)),
        "pred_mean": float(yh.mean()),
    }


if not args.eval_only:
    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(0); t0 = time.time(); step = 0
    for ep in range(args.epochs):
        rm.train()
        wins = sample_train_windows(seqs, args.T, rng)
        for i in range(0, len(wins) - args.bs + 1, args.bs):
            loss, parts = rm.loss(collate(seqs, wins[i:i + args.bs], args.T, True))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0); opt.step(); step += 1
            if step % 100 == 0:
                print(f"ep {ep} step {step} loss {loss.item():.4f} "
                      + " ".join(f"{k} {v.item():.4f}" for k, v in parts.items()) + f" {time.time() - t0:.0f}s", flush=True)
        val = eval_events(seqs.split.t1, seqs.split.t2)
        print(json.dumps({"epoch": ep, "val": val}), flush=True)
        torch.save({"cfg": cfg.to_dict(), "state": rm.state_dict(), "args": vars(args)}, out / "model.pt")
else:
    rm.load_state_dict(torch.load(out / "model.pt")["state"])

res = {"test": eval_events(seqs.split.t2, float("inf")), "oracle": eval_oracle()}
print(json.dumps(res, indent=2), flush=True)
(out / "eval.json").write_text(json.dumps(res, indent=2))
