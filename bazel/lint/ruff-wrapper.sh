#!/bin/bash
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

binary_root=$PWD/..

cd "$BUILD_WORKSPACE_DIRECTORY"

binary=$(find $binary_root -name ruff | head -n 1)

result=0
if [[ $1 == "check" ]]; then
    "$binary" format --check --quiet --diff || result=$?
    "$binary" check --quiet || result=$?
else
    "$binary" format || result=$?
    "$binary" check --fix || result=$?
fi
exit $result
