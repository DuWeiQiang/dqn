#!/bin/sh
set -e

if [ $# -lt 3 ]; then
    echo "usage: evaluate.sh rom_file caffemodel epsilon [save_screen_dir]"
    exit
fi

ROM=$1
MODEL=$2
EPSILON=$3
if [ -z $4 ]; then
    SAVE_SCREEN=""
else
    mkdir -p $4
    SAVE_SCREEN="-save_screen $4"
fi
NET="dqn.prototxt"
NUM_GAMES=1

CMD="dqn -gpu -rom $ROM -evaluate -evaluate_with_epsilon $EPSILON \
    -model $MODEL -net $NET $SAVE_SCREEN -repeat_games $NUM_GAMES"

if [[ `hostname` == *tacc* ]];
then
    cluster --gpu --outfile=eval.out $CMD
else
    $CMD
fi
