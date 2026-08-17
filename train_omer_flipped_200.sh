#!/usr/bin/env bash
set -euo pipefail
cd /home/maxim/slide/HTR-VT
mkdir -p output
/home/maxim/slide/.local/bin/micromamba run -p /home/maxim/slide/.micromamba/envs/htrvt-cpu python -u run_omer_cpu.py --data-dir data/omer_flipped --iters 200 --batch-size 2 --lr 0.001 --out-dir output/omer_flipped_cpu_200
