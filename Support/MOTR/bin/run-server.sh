#!/bin/bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -ex


cd ${MODULAR_PATH}/Support/MOTR/src/cli
../../build/cli/Debug/motr server
