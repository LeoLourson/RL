"""RL-дообучение SFT-политики (RLOO + KL к SFT) по reward model."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import UserSeqs
from rlrec.evaluate import Evaluator
from rlrec.model import Policy, RewardModel
from rlrec.rl import RewardFn, rloo_step, sample_contexts
from transformers import GPT2Config

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/kuairec")
p.add_argument("--sft", default="artifacts/sft/model.pt")
p.add_argument("--rm", default="artifacts/rm/model.pt")
p.add_argument("--out", required=True)
p.add_argument("--reward", default="ratio", choices=["ratio", "time", "cont"])
p.add_argument("--alpha", type=float, default=1.0)
p.add_argument("--lam_rel", type=float, default=0.0)
p.add_argument("--rep_pen", type=float, default=0.0)
p.add_argument("--beta", type=float, default=0.05)
p.add_argument("--K", type=int, default=8)
p.add_argument("--B", type=int, default=64)
p.add_argument("--H", type=int, default=1)
p.add_argument("--gamma", type=float, default=0.9)
p.add_argument("--steps", type=int, default=600)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--eval_every", type=int, default=200)
p.add_argument("--stage", default="val", choices=["val", "test"])
p.add_argument("--threads", type=int, default=16)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed); torch.set_num_threads(args.threads)
rng = np.random.default_rng(args.seed)
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

seqs = UserSeqs(Path(args.data))
ck = torch.load(args.sft)
T = ck["args"]["T"]
policy = Policy(GPT2Config.from_dict(ck["cfg"])); policy.load_state_dict(ck["state"])
ref = copy.deepcopy(policy).eval()
for q in ref.parameters():
    q.requires_grad_(False)
rk = torch.load(args.rm)
rm = RewardModel(GPT2Config.from_dict(rk["cfg"]), torch.from_numpy(seqs.item_dur))
rm.load_state_dict(rk["state"]); rm.eval()
for q in rm.parameters():
    q.requires_grad_(False)
dur_ms = np.load(f"{args.data}/item_dur_ms.npy")
reward_fn = RewardFn(rm, args.reward, dur_ms, args.alpha, args.lam_rel, args.rep_pen)
ev = Evaluator(seqs, args.data, T, args.stage)
opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
log = open(out / "log.jsonl", "w")


def evaluate(step):
    res = {"step": step} | ev.run(policy, rm)
    policy.eval()  # dropout выключен и в RL: иначе KL к ref > 0 уже на шаге 0
    print(json.dumps(res), flush=True); log.write(json.dumps(res) + "\n"); log.flush()
    return res


evaluate(0)
policy.eval(); t0 = time.time()
for step in range(1, args.steps + 1):
    it, rt, hist = sample_contexts(seqs, args.B, T - args.H + 1, rng)
    if args.H > 1:  # место под дописываемые объекты
        it = torch.cat([it, torch.zeros(args.B, args.H - 1, dtype=it.dtype)], 1)
        rt = torch.cat([rt, torch.zeros(args.B, args.H - 1, dtype=rt.dtype)], 1)
    loss, st = rloo_step(policy, ref, reward_fn, it, rt, hist, args.K, args.beta, args.H, args.gamma)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    if step % 20 == 0:
        st |= {"step": step, "t": round(time.time() - t0)}
        print("train " + json.dumps(st), flush=True); log.write(json.dumps({"train": st}) + "\n")
    if step % args.eval_every == 0 or step == args.steps:
        evaluate(step)
torch.save({"cfg": ck["cfg"], "state": policy.state_dict(), "args": vars(args) | {"T": T}}, out / "model.pt")
# итог на test по тому же протоколу, что и для SFT (scripts/eval_test.py)
test = Evaluator(seqs, args.data, T, "test").run(policy, rm)
(out / "test.json").write_text(json.dumps(test | {"args": vars(args)}, indent=2))
print("test " + json.dumps(test), flush=True)
