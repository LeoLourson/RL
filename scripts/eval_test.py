"""Оценка SFT-модели на test по общему протоколу (для сравнения с RL)."""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rlrec.data import UserSeqs
from rlrec.evaluate import Evaluator
from rlrec.model import Policy, RewardModel

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/kuairec")
p.add_argument("--model", default="artifacts/sft/model.pt")
p.add_argument("--rm", default="artifacts/rm/model.pt")
p.add_argument("--out", default="artifacts/sft/test.json")
p.add_argument("--threads", type=int, default=16)
args = p.parse_args()
torch.set_num_threads(args.threads)
seqs = UserSeqs(Path(args.data))
ck = torch.load(args.model)
pol = Policy(GPT2Config.from_dict(ck["cfg"])); pol.load_state_dict(ck["state"])
rk = torch.load(args.rm)
rm = RewardModel(GPT2Config.from_dict(rk["cfg"]), torch.from_numpy(seqs.item_dur)); rm.load_state_dict(rk["state"])
res = Evaluator(seqs, args.data, ck["args"]["T"], "test").run(pol, rm)
Path(args.out).write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
