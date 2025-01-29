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
    --d 1 \
    --m 10 30 100 300 1000 3000 10000 \
    --n 1000 \
    --pool 32 \
    --plot \
    radial_basis \
    --bandwidth 0.1 0.3 0.5

  $PYTHON $SCRIPT \
    --comparator "$cmp" \
    --d 1 \
    --m 5000 \
    --n 10 30 100 300 1000 3000 5000 10000 \
    --pool 32 \
    --plot \
    radial_basis \
    --bandwidth 0.1 0.3 0.5
done
