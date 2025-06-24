#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -e # exit on error
set -x

MOTR_DIR=$(cd "$(dirname "$0")" && cd .. && pwd)

MOTR_BIN=${MOTR_DIR}/build/cli/Release/motr

MOTR_LAYOUTS_DIR=${MOTR_DIR}/src/gui/web/layouts
cd ${MOTR_LAYOUTS_DIR}


PATTERNS=$@
if [[ "$PATTERNS" == "" ]]; then
    PATTERNS=$(cat files.txt)
fi

for pat in $PATTERNS; do
    set -x
    PATTERN=$(ls $pat)
    for f in $PATTERN; do
        if [ -f $f ]; then
            set +x
            ${MOTR_BIN} tags type=set_layout filename=$f "contents=$(cat $f)"
            set -x
        fi
    done
done
