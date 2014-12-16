#!/bin/bash
set -e

if [ $# -lt 1 ]; then
    echo "usage: train.sh rom_file"
    exit
fi

EMAIL="mhauskn@cs.utexas.edu"
ROM=$1
ROM_NAME=`basename $ROM | awk -F'.bin' '{print $1}'`
MAX_ITER=3000000
TACC_ITER_PER_JOB=1000000

if [[ `hostname` == *tacc* ]];
then
    iter=0
    i=0
    while [ $iter -lt $MAX_ITER ]
    do
        iter=$(($iter + $TACC_ITER_PER_JOB))
        if [ -z "$PID" ]; then
            PID=$(cluster --suppress --gpu --outfile ${ROM_NAME}$i.out \
                dqn -gpu -rom $ROM -snapshot_prefix state/$ROM_NAME$i \
                -max_iter $TACC_ITER_PER_JOB)
        else
            PID=$(cluster --depend $PID --suppress --gpu \
                --outfile ${ROM_NAME}$i.out dqn -gpu -rom $ROM \
                -model state/$ROM_NAME$(($i-1))_iter_${TACC_ITER_PER_JOB}.caffemodel \
                -snapshot_prefix state/$ROM_NAME$i \
                -max_iter $TACC_ITER_PER_JOB \
                -memory_threshold 50000 \
                -explore 1)
        fi
        i=$(($i + 1))
        sleep .5
    done
    cluster --suppress --depend $PID --email $EMAIL --gpu echo "$ROM_NAME Done!"
else
    cluster --gpu --email $EMAIL --outfile $ROM_NAME.out \
        dqn -gpu -rom $ROM -snapshot_prefix state/$ROM_NAME
fi
