#!/usr/bin/env bash
set -euo pipefail
cd /home/maxim/HTR-VT
mkdir -p output
/home/maxim/slide/.local/bin/micromamba run -p /home/maxim/slide/.micromamba/envs/htrvt-cpu \
  python -u run_omer_ariel_cpu.py \
  --iters 150 \
  --batch-size 2 \
  --lr 0.0005 \
  --blank-bias -2 \
  --no-final-logit-norm \
  --out-dir output/omer_ariel_fixed_150_lr5e4
