# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #

# Minimal kbench fixture used by unit tests that need an actual .so build.
# Intentionally imports nothing from the kernel libraries so the compilation
# is fast even under --config=ubsan (where the Mojo compiler runs instrumented
# and kernel-dep compilation would otherwise time out the test shard).


def main() raises:
    pass
