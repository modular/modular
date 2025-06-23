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

if [[ "$EXPECT_CRASH" == "1" ]]; then
  exec "$NOT" --crash "$BINARY" "$@" | "$FILECHECK" "$SOURCE"
elif [[ "$EXPECT_FAIL" == "1" ]]; then
  exec "$NOT" "$BINARY" "$@" | "$FILECHECK" "$SOURCE"
else
  exec "$BINARY" "$@" | "$FILECHECK" "$SOURCE"
fi
