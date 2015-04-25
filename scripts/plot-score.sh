#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 log_file [smooth number]"
    exit
fi

if [ -z "$2" ]; then
    SMOOTH=1
else
    SMOOTH=$2
fi
grep score $1 | lmj-plot -m 'score = (\S+),.*' --xlabel Episode --ylabel Score --title $1 -g -T -s $SMOOTH &
grep Iteration $1 | lmj-plot -m 'Iteration (\d+), loss = (\S+)' --num-x-ticks 8 --xlabel Iteration --ylabel Loss --title $1 -g -T &
# grep lr $1 | lmj-plot -m 'Iteration (\d+), lr = (\S+)' --num-x-ticks 8 --xlabel Iteration --ylabel lr --title $1 -g -T &
