#!/bin/sh
set -e

if [ $# -lt 3 ]; then
    echo "usage: create_movie.sh rom_file caffemodel output_movie"
    exit
fi

ROM=$1
MODEL=$2
MOVIE=$3
NET="dqn.prototxt"
SCREEN_DIR="screens"
FRAMERATE=25

echo "==> Using directory \"$SCREEN_DIR\" for png files"
sleep .5
mkdir -p $SCREEN_DIR

echo "==> Evaluating $MODEL on $ROM"
sleep .5
./dqn -evaluate -rom $ROM -save_screen $SCREEN_DIR/ -model $MODEL -net $NET

echo "==> Converting pngs into movie: $MOVIE"
sleep .5
ffmpeg -framerate $FRAMERATE -i $SCREEN_DIR/%05d.png -r 30 -pix_fmt yuv420p $MOVIE

echo "==> Cleaning up"
sleep .5
rm $SCREEN_DIR/*
rmdir $SCREEN_DIR

echo "==> Done!"
