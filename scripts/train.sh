#!/bin/bash
set -e

if [ $# -lt 2 ]; then
    echo "usage: train.sh rom_file exp_dir"
    exit
fi

EMAIL="mhauskn@cs.utexas.edu"
ROM=$1
EXP_DIR=$2
ROM_NAME=`basename $ROM | awk -F'.bin' '{print $1}'`

# Create exp_dir
mkdir -p $EXP_DIR

if [[ `hostname` == *tacc* ]];
then
    MAX_ITER=10000000
    TACC_ITER_PER_JOB=1000000
    MEMORY_THRESHOLD=100000
    iter=0
    i=0
    while [ $iter -lt $MAX_ITER ]
    do
        if [ -z "$PID" ]; then
            PID=$(cluster --suppress --gpu --outfile $EXP_DIR/${ROM_NAME}$i.out \
                dqn -gpu -rom $ROM -snapshot_prefix $EXP_DIR/$ROM_NAME \
                -max_iter $TACC_ITER_PER_JOB)
        else
            PID=$(cluster --depend $PID --suppress --gpu \
                --outfile $EXP_DIR/${ROM_NAME}$i.out dqn -gpu -rom $ROM \
                -snapshot ${EXP_DIR}/${ROM_NAME}_iter_${iter}.solverstate \
                -snapshot_prefix $EXP_DIR/$ROM_NAME \
                -max_iter $TACC_ITER_PER_JOB \
                -memory_threshold $MEMORY_THRESHOLD \
                -explore 1)
        fi
        iter=$(($iter + $TACC_ITER_PER_JOB))
        i=$(($i + 1))
        sleep .5
    done
    cluster --suppress --depend $PID --email $EMAIL --gpu echo "$ROM_NAME Done!"
else
    cluster --suppress --gpu --email $EMAIL --outfile $EXP_DIR/$ROM_NAME.out dqn -gpu -rom $ROM -snapshot_prefix $EXP_DIR/$ROM_NAME
fi
