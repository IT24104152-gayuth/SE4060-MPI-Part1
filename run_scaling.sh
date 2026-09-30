#!/bin/bash
# run_scaling.sh - Exercise 4 data collection
# Runs ex02_sum and ex03_pi for 1, 2, 4, 8, 16 processes, repeats each 5 times
# and keeps the best (minimum) wall time. Writes sum_scaling.csv and
# pi_scaling.csv with columns: processes,time,speedup
#
# Usage: ./run_scaling.sh
#        PROCS="1 2 4" REPEATS=3 ./run_scaling.sh
#        N=2000000000 DARTS=100000000 ./run_scaling.sh   (bigger problem size if
#        the default 10,000,000 finishes too fast to plot a clean curve)

set -u

PROCS=${PROCS:-"1 2 4 8 16"}
REPEATS=${REPEATS:-5}
N=${N:-10000000}          # Exercise 2 problem size
DARTS=${DARTS:-10000000}  # Exercise 3 problem size
OVERSUB="--oversubscribe"      # needed when ranks > physical cores
# For a real multi-node run add: HOSTFILE=hosts ./run_scaling.sh
HOSTS=${HOSTFILE:+--hostfile $HOSTFILE}

run_one() {
    local exe=$1 np=$2 arg=$3
    local best=""
    for ((r = 0; r < REPEATS; r++)); do
        local t
        t=$(mpirun -np "$np" $OVERSUB ${HOSTS:-} "./$exe" "$arg" 2>/dev/null \
            | grep '^CSV,' | cut -d, -f3)
        [ -z "$t" ] && continue
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best=$t; fi
    done
    echo "$best"
}

collect() {
    local exe=$1 out=$2 label=$3 arg=$4
    echo "== $label =="
    local base=""
    : > "$out.tmp"
    for np in $PROCS; do
        local t
        t=$(run_one "$exe" "$np" "$arg")
        if [ -z "$t" ]; then echo "  np=$np FAILED"; continue; fi
        [ -z "$base" ] && base=$t
        local sp
        sp=$(awk "BEGIN{printf \"%.4f\", $base / $t}")
        printf "  np=%-3s time=%-10s speedup=%s\n" "$np" "$t" "$sp"
        echo "$np,$t,$sp" >> "$out.tmp"
    done
    { echo "processes,time,speedup"; cat "$out.tmp"; } > "$out"
    rm -f "$out.tmp"
    echo "  -> $out"
}

collect ex02_sum sum_scaling.csv "Exercise 2 - sum 1..$N" "$N"
collect ex03_pi   pi_scaling.csv  "Exercise 3 - Monte Carlo Pi, $DARTS darts" "$DARTS"
