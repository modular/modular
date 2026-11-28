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
# C reference: c_abi_test_tail_allocas.c
# RUN: mkdir -p %t.dir
# RUN: mojo build -O1 -Xlinker $(dirname %s)/libc_abi_reference.lo %s -o %t.dir/test_tail_allocas
# RUN: %t.dir/test_tail_allocas | FileCheck %s
#
# An `abi("C")` function that passes a large by-value struct to another Mojo
# function, or returns a large struct that another Mojo function made, moves
# the struct through a local temporary. The call receives a pointer to that
# temporary, so it must not be marked `tail`: at -O1 LLVM would delete the
# store into the temporary or read it as undef.

from std.collections import Array
from std.ffi import external_call


@fieldwise_init
struct TailBig(Copyable):
    var v: Array[Int64, 64]


@no_inline
def total(b: TailBig) -> Int64:
    var s: Int64 = 0
    for i in range(64):
        s += b.v[i]
    return s


def c_tail_weigh_param(b: TailBig) abi("C") -> Int64:
    return total(b)


@no_inline
def make(n: Int64) -> TailBig:
    return TailBig(Array[Int64, 64](fill=n))


def c_tail_weigh_result(n: Int64) abi("C") -> TailBig:
    return make(n)


def main():
    external_call["c_tail_drive_param", NoneType](c_tail_weigh_param)
    external_call["c_tail_drive_result", NoneType](c_tail_weigh_result)


# CHECK: sum = 448
# CHECK: v[0] = 7, v[63] = 7
