#!/bin/sh

# Set source
basedir=/scratch/cluster/ajh/neural_scene_representation_and_rendering
. ${basedir}/env/bin/activate

python3 $basedir/train.py

