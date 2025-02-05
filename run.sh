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
  regression_rotation
)
# Since tuning is toroidal in [-1, 1] and bandwidth sets the tuning=0.5 distance, the 'max separation'
# of two neurons is 1, so a max bandwidth of 0.5 (in 1D) is reasonable. Then, scale up bandwidths
# by sqrt(dim)
BANDWIDTHS_1D="0.1 0.3 0.5"
BANDWIDTHS_2D="0.1414213562 0.4242640687 0.7071067812"

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
    --bandwidth "$BANDWIDTHS_2D"

  $PYTHON $SCRIPT \
    --comparator "$cmp" \
    --poisson-scale 0.0 \
    --d 1 --extra-dims 1 \
    --m 5000 \
    --n 10 30 100 300 1000 3000 \
    --pool 8 \
    --plot \
    radial_basis \
    --bandwidth "$BANDWIDTHS_2D"
done
