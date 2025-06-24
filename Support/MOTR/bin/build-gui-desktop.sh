#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -ex
SCRIPT_DIR=$(cd $(dirname "$0") && pwd)
cd $SCRIPT_DIR

SRC_DIR=${SRC_DIR:-${SCRIPT_DIR}/src/gui}
BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/build/gui/desktop}
IMGUI_DAWN_DIR=${IMGUI_DAWN_DIR:-${SCRIPT_DIR}/src/third-party/dawn}
CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}

function checkout_dawn() {
    mkdir -p ${IMGUI_DAWN_DIR}
    cd ${IMGUI_DAWN_DIR}
    git clone --depth 1 https://github.com/google/dawn.git
    cd dawn
    git checkout 539ed5c222e1059ad61ccef77e139025f3239eda
}

checkout_dawn

if [[ ! -z ${CLEAN} ]]; then
    rm -rf ${BUILD_DIR}
fi

if [[ ! -d ${BUILD_DIR} ]]; then
    mkdir -p ${BUILD_DIR}
    cmake \
        -G Ninja \
        -S ${SRC_DIR} \
        -B ${BUILD_DIR} \
        -DIMGUI_DAWN_DIR=${IMGUI_DAWN_DIR} \
        -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE
fi

cd ${BUILD_DIR}

ninja -j12 -v
