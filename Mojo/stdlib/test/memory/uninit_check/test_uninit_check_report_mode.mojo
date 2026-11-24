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
# Tests `MOJO_STDLIB_SIMD_UNINIT_CHECK=report`: a poison match reports and
# execution continues, so one run can enumerate every offending site rather
# than stopping at the first. Two distinct poisoned loads, then a line that
# only prints if neither aborted.

from std.memory import Pointer


# CHECK: UNINIT_READ at {{.*}}: dtype={{.*}}: load matched debug allocator poison sentinel
# CHECK: UNINIT_READ at {{.*}}: dtype={{.*}}: load matched debug allocator poison sentinel
# CHECK: SURVIVED
def main():
    var first = UInt32(0x7F7FFFFF)
    _ = Pointer(to=first).unsafe_bitcast[Float32]().unsafe_load()

    var second = UInt64(0x7FEFFFFFFFFFFFFF)
    _ = Pointer(to=second).unsafe_bitcast[Float64]().unsafe_load()

    print("SURVIVED")
