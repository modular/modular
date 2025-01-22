#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##
#
# This file is a helper script for pre-commit to run prettier correctly
# within the VS Code extension from the root of the repository.
#
##===----------------------------------------------------------------------===##

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR/../"
npx prettier --write .
