#!/bin/bash

if [ $# -ne 1 ]; 
    then echo "You need to provide the defs file as an input argument!"
    exit
fi

# Load defs
source $1

if [ $USESCRIPT -eq 1 ]; then
    echo "################### TRAIN START (LOGGING) ##############"
    script -f -c "$TOOLS/caffe train\
    --solver=${SOLVERNAME}"\
    ${TRAINLOG}
    echo " "
    echo "################### TRAIN STOPS (LOGGED ) ##############"
else
    echo "################### TRAIN START (NO LOG) ##############"
    $TOOLS/caffe train\
    --solver=${SOLVERNAME}
    echo " "
    echo "################### TRAIN STOPS (NO LOG) ##############"
fi;

