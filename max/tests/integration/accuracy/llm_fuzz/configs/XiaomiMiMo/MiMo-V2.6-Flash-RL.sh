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

# The artifact's export envelope: batch 8 at the 16384-token limit.
batch_size=8
max_length=16384

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=mimo_v2
scenarios=
k2vv_mode=
circuit_breaker=0

# Deliberately empty: no known-failure exemptions yet. A failing test is a
# bug to fix, not an entry to add here.
exclude=
