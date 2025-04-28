#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT=$(realpath "${SCRIPT_DIR}/../..")
BUILD_DIR="${REPO_ROOT}"/build
mkdir -p "${BUILD_DIR}"

STDLIB_PATH="${REPO_ROOT}/stdlib/src"

echo "Packaging up the Standard Library."
STDLIB_PACKAGE_NAME="stdlib.mojopkg"
FULL_STDLIB_PACKAGE_PATH="${BUILD_DIR}"/"${STDLIB_PACKAGE_NAME}"
mojo package "${STDLIB_PATH}" -o "${FULL_STDLIB_PACKAGE_PATH}"

# The `mojo` nightly compiler ships with its own `stdlib.mojopkg`. For the
# open-source stdlib, we need to specify the paths to the just-built
# `stdlib.mojopkg`. Otherwise, without this, the `mojo` compiler would
# use its own `stdlib.mojopkg` it ships with which is not what we want.
export MODULAR_MOJO_MAX_IMPORT_PATH=$BUILD_DIR

echo Successfully created "${FULL_STDLIB_PACKAGE_PATH}"
