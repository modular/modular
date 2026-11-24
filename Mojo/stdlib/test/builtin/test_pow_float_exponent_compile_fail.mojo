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

# Casting the exponent is the wrong half to cast: the result takes the dtype
# of the base, so an integral base with a floating exponent is rejected and
# the diagnostic points at casting the base instead.


# CHECK: an integer cannot be raised to a floating point power
def main():
    var a: Int = 2
    var p: Float64 = -1.0
    print(a**p)
