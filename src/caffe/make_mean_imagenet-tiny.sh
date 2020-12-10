#!/bin/bash
# Compute the mean image from the imagenet training leveldb
# N.B. this is available in data/ilsvrc12

if [ $# -ne 1 ]; 
    then echo "You need to provide the defs file as an input argument!"
    return -1
fi

# Load defs
source $1

${TOOLS}/compute_image_mean.bin $TRAINDB $MEANFILE

echo "Done."
