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


# Indexing a SIMD vector by a raised loop's induction variable. Raising the
# loop lets SCCP explore the induction variable's exit value, one past the
# vector, so the element folders must decline that position instead of
# indexing out of bounds. Regression test: this used to abort the compiler.

# RUN: kgen --emit=llvm %s -o %t.ll
# RUN: FileCheck %s < %t.ll


@export
def fill_simd() -> SIMD[.uint8, 4]:
    var v = SIMD[.uint8, 4](0)
    for i in range(4):
        v[i] = UInt8(i)
    return v


@export
def read_simd(v: SIMD[.int32, 8]) -> Int:
    var total = 0
    for i in range(8):
        total = total + Int(v[i])
    return total


# CHECK-LABEL: @fill_simd
# CHECK-LABEL: @read_simd
