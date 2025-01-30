#!/usr/bin/env bash

PYTHON=/home/rdlvcs/.virtualenvs/reps-bias-variance/bin/python
SCRIPT=/home/rdlvcs/Research/reps-bias-variance/run.py
COMPARATORS=(
  procrustes
  linear_cka
  debiased_linear_cka
  brownian_cka
  debiased_brownian_cka
  regression
)

for cmp in "${COMPARATORS[@]}"; do
  $PYTHON $SCRIPT \
    --comparator "$cmp" \
    --poisson-scale 0.0 \
    --d 1 --extra-dims 1 \
    --m 10 30 100 300 1000 3000 10000 \
    --n 1000 \
    --pool 8 \
    --plot \
    radial_basis \
    --bandwidth 0.15 0.45 0.75

  $PYTHON $SCRIPT \
    --comparator "$cmp" \
    --poisson-scale 0.0 \
    --d 1 --extra-dims 1 \
    --m 5000 \
    --n 10 30 100 300 1000 3000 \
    --pool 8 \
    --plot \
    radial_basis \
    --bandwidth 0.15 0.45 0.75
done
