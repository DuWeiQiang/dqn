#!/bin/sh
set -e

if [ $# -lt 1 ]; then
    echo "usage: train.sh rom_file"
    exit
fi

EMAIL="mhauskn@cs.utexas.edu"
ROM=$1
ROM_NAME=`basename $ROM | head -c 4`

echo cluster --gpu --email $EMAIL --outfile $ROM_NAME.out \
    dqn -gpu -rom $ROM -snapshot_prefix state/$ROM_NAME
