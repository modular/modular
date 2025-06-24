#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -ex
SCRIPT_DIR=$(cd "$(dirname "$0")" && cd .. && pwd)
cd $SCRIPT_DIR

SRC_DIR=${SRC_DIR:-${SCRIPT_DIR}/src/test}

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}
BUILD_SUBDIR=${BUILD_SUBDIR:-${CMAKE_BUILD_TYPE}}
BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/build/test/${BUILD_SUBDIR}}


if [[ ! -z ${CLEAN} ]]; then
    rm -rf ${BUILD_DIR}
fi

if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir -p ${BUILD_DIR}
    # ln -sf ${BUILD_DIR}/compile_commands.json ${SCRIPT_DIR}/compile_commands.json

    cmake \
        -G Ninja \
        -S ${SRC_DIR} \
        -B ${BUILD_DIR} \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
fi

cd ${BUILD_DIR}

ninja -j12 -v

./motr-test
