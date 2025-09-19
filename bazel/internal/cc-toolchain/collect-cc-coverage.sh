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

# This script is based on and replaces https://github.com/bazelbuild/bazel/blob/3f42ff95d59921ed0bb96258e191bfee58fed554/tools/test/collect_cc_coverage.sh

if [[ -n "${MODULAR_VERBOSE_COVERAGE:-}" ]]; then
  set -x
fi

# Make sure running llvm-profdata doesn't produce its own profile.
unset LLVM_PROFILE_FILE
llvm_profdata=("$ROOT"/external/+http_archive+*/bin/llvm-profdata)
llvm_cov=("$ROOT"/external/+http_archive+*/bin/llvm-cov)

if [[ ! -x "$llvm_profdata" || ! -x "$llvm_cov" ]]; then
  echo "error: llvm-profdata or llvm-cov not found" >&2
  ls ./external
  exit 1
fi

readonly profdata="$COVERAGE_DIR/coverage.pd" # NOTE: .profdata extension cannot be used because bazel assumes you will have either .profdata or .dat (lcov), not both
readonly html_output="$COVERAGE_DIR/html"

rm -f "$profdata"
mkdir -p "$html_output"

"$llvm_profdata" merge -sparse -output "$profdata" "$COVERAGE_DIR"/*.profraw

object_param=""
while read -r line; do
  if [[ ${line: -24} == "runtime_objects_list.txt" ]]; then
    while read -r line_runtime_object; do
      if [[ "$line_runtime_object" == */llvm-symbolizer ]]; then
        continue  # Skip because it is not built for coverage
      fi

      # NOTE: Some llvm-cov commands expect at least 1 object to be specified directly.
      if [[ -z "$object_param" ]]; then
        object_param="$ROOT/$line_runtime_object"
      else
        object_param+=" -object $ROOT/$line_runtime_object"
      fi
    done < "$ROOT/$line"
  fi
done < "$COVERAGE_MANIFEST"

llvm_cov_args=(
  $object_param
  -ignore-filename-regex='^/tmp/.+'
  -ignore-filename-regex='external/.+'
  -instr-profile "$profdata"
  -path-equivalence .,$TEST_SRCDIR/_main
)

# Parse-able lcov file
"$llvm_cov" export -format=lcov "${llvm_cov_args[@]}" > "$COVERAGE_DIR/_cc_coverage.dat"

# ASCII coverage representation
"$llvm_cov" show "${llvm_cov_args[@]}" > "$COVERAGE_DIR/coverage.txt"

# HTML coverage report
"$llvm_cov" show -format=html "${llvm_cov_args[@]}" --output-dir="$html_output"
pushd "$COVERAGE_DIR" > /dev/null
tar czf html.tar.gz html
popd > /dev/null
