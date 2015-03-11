#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 log_file"
    exit
fi
grep score $1 | lmj-plot -m 'score = (\S+),.*' --xlabel Episode --ylabel Score --title $1 -g -T &
grep Iteration $1 | lmj-plot -m 'Iteration (\d+), loss = (\S+)' --num-x-ticks 8 --xlabel Iteration --ylabel Loss --title $1 -g -T &
