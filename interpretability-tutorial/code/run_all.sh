#!/bin/sh
# Runs every experiment of the notes, one after the other (about half an hour on a laptop CPU).
# Pass --replot to reuse a finished training run of the last session.
set -e
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
for script in exp_basics exp_ioi exp_directions exp_neurons exp_superposition exp_sae exp_induction; do
    "$PYTHON" "$script.py"
done
"$PYTHON" exp_modadd.py "$@"
