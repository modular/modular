##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

# shellcheck disable=SC2034  # Variables are used when sourced

# Fuzzed on Mach only (no MAX Serve lane). The known-failures list was
# measured at this context length.
max_length=32768

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=muse_spark
scenarios=
k2vv_mode=
circuit_breaker=0

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/muse-spark-known-failures.sh"
