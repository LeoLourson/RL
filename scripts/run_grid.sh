#!/usr/bin/env bash
# Сетка RL-экспериментов (последовательно, 16 потоков; шаг ~0,4 с).
# lr=1e-4 при β=0,05 за 60 шагов схлопывает политику в один ролик для всех — см. README.
set -u
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python
STEPS=${STEPS:-300}

run() {  # имя, затем аргументы train_rl.py
  local name=$1; shift
  $PY -u scripts/train_rl.py --out "artifacts/rl/$name" --steps "$STEPS" --lr 1e-5 --eval_every 100 "$@" \
    > "artifacts/rl_$name.log" 2>&1
}
mkdir -p artifacts/rl

run ratio_b0.1 --reward ratio --beta 0.1
run ratio_b0.3 --reward ratio --beta 0.3
run ratio_b1.0 --reward ratio --beta 1.0
run time_b0.3 --reward time --beta 0.3
run cont_b0.3 --reward cont --alpha 1.0 --beta 0.3
run ratio_rep_b0.3 --reward ratio --rep_pen 0.5 --beta 0.3
run ratio_rel_b0.3 --reward ratio --lam_rel 0.1 --beta 0.3
run ratio_H3_b0.3 --reward ratio --H 3 --beta 0.3
