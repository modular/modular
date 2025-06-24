#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -ex
SCRIPT_DIR=$(cd "$(dirname "$0")" && cd .. && pwd)
cd $SCRIPT_DIR

MODULAR_SRC_DIR=$(cd $SCRIPT_DIR/../../.. && pwd)

cd ${MODULAR_SRC_DIR}

source utils/start-modular.sh

BAZEL=${MODULAR_SRC_DIR}/bazelw

TARGET="//Support/tools/build-info"

${BAZEL} build ${TARGET}

TARGET_EXE=$PWD/$(${BAZEL} cquery ${TARGET} --output=files | head -1)

iteration=0
set +x
while true; do
    iteration=$((iteration + 1))
    echo
    date
    echo "# [$iteration] running 2 instances of ${TARGET_EXE}..."
    ${TARGET_EXE} &
    ${TARGET_EXE} &
    echo "# sleep 10s..."
    sleep 10
done