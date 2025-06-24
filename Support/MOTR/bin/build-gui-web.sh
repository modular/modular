#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -ex
SCRIPT_DIR=$(cd "$(dirname "$0")" && cd .. && pwd)
cd $SCRIPT_DIR

# emsdk 4.0.8 (https://github.com/emscripten-core/emsdk/tree/4.0.8)
EMSDK_GIT_TAG=419021fa040428bc69ef1559b325addb8e10211f

SRC_DIR=${SRC_DIR:-${SCRIPT_DIR}/src/gui}

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Debug}
BUILD_SUBDIR=${BUILD_SUBDIR:-${CMAKE_BUILD_TYPE}}
BUILD_DIR=${BUILD_DIR:-${SCRIPT_DIR}/build/gui/web/${BUILD_SUBDIR}}

# need to make BUILD_DIR exists to canoncialize it and EMSDK_REPO_DIR
if [[ -d ${BUILD_DIR} ]]; then
    build_dir_already_exists=true
else
    mkdir -p ${BUILD_DIR}
fi

BUILD_DIR=$(cd "${BUILD_DIR}" && pwd) # canonicalize

EMSDK_REPO_DIR=${EMSDK_REPO_DIR:-${BUILD_DIR}/../emsdk}
mkdir -p ${EMSDK_REPO_DIR}
EMSDK_REPO_DIR=$(cd "${EMSDK_REPO_DIR}" && pwd) # canonicalize

function fetch_emsdk() {
    if [[ ! -d "${EMSDK_REPO_DIR}/.git" ]]; then
        echo "Fetching emsdk ${EMSDK_GIT_TAG}"
        mkdir -p "${EMSDK_REPO_DIR}"
        cd "${EMSDK_REPO_DIR}"
        git init
        git remote add origin https://github.com/emscripten-core/emsdk.git
        git fetch --depth=1 origin ${EMSDK_GIT_TAG}
        git checkout FETCH_HEAD
    fi
    cd "${EMSDK_REPO_DIR}"
    REPO_HASH=$(git rev-parse HEAD)
    echo "emsdk repo hash: ${REPO_HASH}"
    if [[ ! ${REPO_HASH} = ${EMSDK_GIT_TAG} ]]; then
        echo "emsdk repo hash does not match expected tag"
        exit 1
    fi
}

function activate_emsdk() {
    fetch_emsdk
    cd "${EMSDK_REPO_DIR}"
    ./emsdk install latest
    ./emsdk activate latest
    source ./emsdk_env.sh
}


if [[ ! -z ${CLEAN} ]]; then
    rm -rf ${BUILD_DIR}
fi

activate_emsdk

if [[ ! ${build_dir_already_exists} ]]; then
    # Configure with emscripten toolchain via emcmake
    emcmake cmake \
        -G Ninja \
        -S ${SRC_DIR} \
        -B ${BUILD_DIR} \
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}

    ln -sf ${BUILD_DIR}/compile_commands.json ${SCRIPT_DIR}/compile_commands.json
fi

cd ${BUILD_DIR}

ninja -j12 -v
