#!/bin/sh
ARCH=$1
INITDIR=$2
if [ $# -ne 2 ]
  then
    echo "Arguments error: <ARCH> <INITDIR>"
    exit 1
fi

python -u examples/cluster.py -d pitts -a ${ARCH} -b 64 --width 640 --height 480 --init-dir ${INITDIR}
