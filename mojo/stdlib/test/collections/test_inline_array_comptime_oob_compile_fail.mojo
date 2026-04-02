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

# RUN: not %mojo %s 2>&1 | FileCheck %s

# CHECK: InlineArray index out of bounds during compile-time evaluation
# CHECK-NOT: failed to run the pass manager
def oob_at_comptime[size: Int](points: InlineArray[Float32, size]) -> Float32:
    comptime for i in range(size):
        comptime for _r in range(1, size):
            _ = points[i]
            _ = points[i + 1]
    return points[0]


def main():
    var pts: InlineArray[Float32, 4] = [0.0, 1.0, 2.0, 3.0]
    _ = oob_at_comptime(pts)