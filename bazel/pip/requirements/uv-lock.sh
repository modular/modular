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

python="$PWD/{{python}}"

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

ln -s "$PWD/{{pyproject}}" "$tmpdir/pyproject.toml"
cp "$PWD/{{existing_lockfile}}" "$tmpdir/uv.lock"

"{{uv}}" \
  lock \
  --directory "$tmpdir" \
  --python "$python" \
  --quiet

cp "$tmpdir/uv.lock" "{{lockfile_output}}"
