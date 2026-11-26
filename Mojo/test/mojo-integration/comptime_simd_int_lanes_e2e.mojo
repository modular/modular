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

# RUN: %mojo %s | FileCheck %s

# Building a `SIMD[DType.int, N]` or `SIMD[DType.uint, N]` from separate lanes
# must work at compile time.


def main():
    # CHECK: lanes: [3, -4] [0, 7]
    comptime v = SIMD[DType.int, 2](3, -4)
    comptime u = SIMD[DType.uint, 2](0, 7)
    print("lanes:", v, u)
