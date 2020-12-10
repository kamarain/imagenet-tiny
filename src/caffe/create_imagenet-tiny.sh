#!/bin/bash
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs
# NOTE: treats all images as RGB since -gray not provided to convert_imageset

if [ $# -ne 1 ]; 
    then echo "You need to provide the defs file as an input argument!"
    return -1
fi

# Load defs
source $1

# Make tempwork
mkdir -p ${TEMPWORK}

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    ${CONVERT_IMAGESET_ARGS} \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    ${TRAINDB}


echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    ${CONVERT_IMAGESET_ARGS} \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    ${VALDB}

echo "Done."
