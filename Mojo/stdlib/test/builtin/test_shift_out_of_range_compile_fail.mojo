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


# CHECK: shift amount 9 is outside the valid range 0..7 for {{UInt8|uint8|scalar<ui8>}}
# CHECK-NOT: could not simplify operator shr(1, 9)
def main():
    # Force the parameter evaluator to materialize the shift result instead of
    # silently discarding the expression.
    var x = UInt8(1) >> 9
    _ = x
