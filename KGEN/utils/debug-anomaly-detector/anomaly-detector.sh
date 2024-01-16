#!/bin/sh
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

"$MODULAR_DERIVED_PATH/autovenv/bin/python3" "$SCRIPT_DIR/anomaly-detector.py" "$@"
