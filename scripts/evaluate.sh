#!/bin/sh
set -e

if [ $# -lt 3 ]; then
    echo "usage: evaluate.sh rom_file caffemodel epsilon"
    exit
fi

ROM=$1
MODEL=$2
EPSILON=$3

./dqn -gui -rom $ROM -evaluate -evaluate_with_epsilon $EPSILON -model $MODEL
