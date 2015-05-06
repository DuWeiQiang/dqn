#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: $0 log_files [smooth number]"
    exit
fi

# if [ -z "$2" ]; then
#     SMOOTH=1
# else
#     SMOOTH=$2
# fi
LOGS=""
for var in "$@"
do
    LOGS+="$var "
done

grep score $LOGS | lmj-plot -m 'score = (\S+),.*' --xlabel Episode --ylabel Score --title $1 -g -T &
grep Iteration $LOGS | lmj-plot -m 'Iteration (\d+), loss = (\S+)' --num-x-ticks 8 --xlabel Iteration --ylabel Loss --title $1 -g -T &
