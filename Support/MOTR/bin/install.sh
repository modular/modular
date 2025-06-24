#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -ex
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd $SCRIPT_DIR

MOTR_DIR=${MOTR_DIR:-$(cd .. && pwd)}
INSTALL_DIR=${INSTALL_DIR:-${MOTR_DIR}/install}

BUILD=${BUILD:-1}
DEBUG=${DEBUG:-1}


# Install motr server cli
function install_motr_cli() {
    SRC_DIR=${MOTR_DIR}/src/cli
    BUILD_DIR=${MOTR_DIR}/build/cli/${1}
    if [[ ! -d "${BUILD_DIR}" ]]; then
        echo "Build directory ${BUILD_DIR} does not exist"
        return
    fi

    if [[ "${1}" == "Debug" ]]; then
        DST_INSTALL_DIR=${INSTALL_DIR}/bin/Debug
    else
        DST_INSTALL_DIR=${INSTALL_DIR}/bin/
    fi

    mkdir -p ${DST_INSTALL_DIR}
    cp ${BUILD_DIR}/motr ${DST_INSTALL_DIR}
    cp ${SRC_DIR}/config.install.${1}.yaml ${DST_INSTALL_DIR}/config.yaml
}

# install_motr_cli Release

function install_motr_include() {
    mkdir -p ${INSTALL_DIR}/include
    rsync -avP -L ${MOTR_DIR}/src/common/include ${INSTALL_DIR}/include
}

function install_motr_web() {
    BUILD_DIR=${MOTR_DIR}/build/gui/web/${1}
    if [ ! -d "${BUILD_DIR}" ]; then
        echo "Build directory ${BUILD_DIR} does not exist"
        return
    fi

    DST_INSTALL_DIR=${INSTALL_DIR}/web
    [[ "${1}" == "Debug" ]] && DST_INSTALL_DIR=${DST_INSTALL_DIR}/Debug

    mkdir -p ${DST_INSTALL_DIR}

    files=$(ls "${BUILD_DIR}/motr_gui"* "${BUILD_DIR}/index"* "${BUILD_DIR}/favicon.ico")
    rsync -avP -L ${files} ${DST_INSTALL_DIR}

    cd ${DST_INSTALL_DIR}
    chmod a-x *.wasm
    # gzip *
}

if [[ "${DEBUG}" == 1 ]]; then
    build_types="Debug Release"
else
    build_types="Release"
fi


ROOT_BUILD_DIR=${MOTR_DIR}/build
if [[ "${CLEAN}" == 1 && -e ${ROOT_BUILD_DIR} ]]; then
    echo "backing up ${ROOT_BUILD_DIR} to ${ROOT_BUILD_DIR}.bak"
    rm -rf ${ROOT_BUILD_DIR}.bak
    mv ${ROOT_BUILD_DIR} ${ROOT_BUILD_DIR}.bak
fi

if [[ "${BUILD}" == 1 ]]; then
    for build_type in ${build_types}; do
        CMAKE_BUILD_TYPE=${build_type} ${SCRIPT_DIR}/build-cli.sh
        CMAKE_BUILD_TYPE=${build_type} ${SCRIPT_DIR}/build-gui-web.sh
    done
fi

if [[ "${CLEAN}" == 1 && -e ${INSTALL_DIR} ]]; then
    echo "backing up ${INSTALL_DIR} to ${INSTALL_DIR}.bak"
    rm -rf ${INSTALL_DIR}.bak
    mv ${INSTALL_DIR} ${INSTALL_DIR}.bak
fi


for build_type in ${build_types}; do
    install_motr_cli ${build_type}
    install_motr_web ${build_type}
done

find ${INSTALL_DIR} -type f

[[ -e ${INSTALL_DIR}/bin/motr ]] && ${INSTALL_DIR}/bin/motr --version || true
[[ -e ${INSTALL_DIR}/bin/Debug/motr ]] && ${INSTALL_DIR}/bin/Debug/motr --version || true
