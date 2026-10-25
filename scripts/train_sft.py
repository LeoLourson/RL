"""Базовая модель: GPT-2, next item prediction на train-части."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import UserSeqs, collate, sample_train_windows
from rlrec.evaluate import Evaluator
from rlrec.model import Policy, gpt2_config

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/kuairec")
p.add_argument("--out", default="artifacts/sft")
p.add_argument("--T", type=int, default=100)
p.add_argument("--d", type=int, default=128)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--heads", type=int, default=2)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--threads", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
rng = np.random.default_rng(args.seed)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
seqs = UserSeqs(Path(args.data))
cfg = gpt2_config(seqs.n_items, args.d, args.layers, args.heads, args.T)
model = Policy(cfg)
ev = Evaluator(seqs, args.data, args.T, "val")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total = sum(len(sample_train_windows(seqs, args.T, np.random.default_rng(1))) // args.bs for _ in [0]) * args.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total + 10, pct_start=0.05)
log = open(out / "log.jsonl", "w")

step, t0 = 0, time.time()
for ep in range(args.epochs):
    model.train()
    wins = sample_train_windows(seqs, args.T, rng)
    for i in range(0, len(wins) - args.bs + 1, args.bs):
        loss = model.sft_loss(collate(seqs, wins[i:i + args.bs], args.T)["items"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % 100 == 0:
            print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
    res = {"epoch": ep, "step": step, "time_s": time.time() - t0} | ev.run(model)
    print(json.dumps(res), flush=True); log.write(json.dumps(res) + "\n"); log.flush()
    torch.save({"cfg": cfg.to_dict(), "state": model.state_dict(), "args": vars(args)}, out / "model.pt")
