#!/bin/bash

if [ $# -ne 1 ]; 
    then echo "You need to provide the defs file as an input argument!"
    return -1
fi

# Load defs
source $1

if [ $USESCRIPT -eq 1 ]; then
    echo "################### TRAIN START (LOGGING) ##############"
    script -a -f -c "$TOOLS/caffe train \
    --solver=${SOLVERNAME} \
    --snapshot=${SNAPSHOTPREFIX}_iter_${RESUMEITER}.solverstate" \
    ${TRAINLOG}
    echo " "
    echo "################### TRAIN STOPS (LOGGED ) ##############"
else
    echo "################### TRAIN START (NO LOG) ##############"
    $TOOLS/caffe train \
    --solver=${SOLVERNAME} \
    --snapshot=${SNAPSHOTPREFIX}_iter_${RESUMEITER}.solverstate
    echo " "
    echo "################### TRAIN STOPS (NO LOG) ##############"
fi;
