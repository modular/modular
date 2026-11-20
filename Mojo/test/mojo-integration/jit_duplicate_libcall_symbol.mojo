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
# `mojo run` splits the host module into one codegen unit per function and
# merges the units into one object. Each unit that lowers a runtime-sized
# `llvm.memset` to a `memset` libcall creates its own `_memset` symbol. The
# merged object must hold one undefined entry per name: ORC JITLink aborts on
# a duplicate.

# REQUIRES: system-darwin
# RUN: %mojo -O0 %s

from std.memory import stack_allocation
from std.sys.intrinsics import llvm_intrinsic


def zero_range_a(p: MutPointer[Byte, ...], count: Int):
    llvm_intrinsic["llvm.memset", NoneType](
        p.unsafe_bitcast[NoneType](), UInt8(0), count, False
    )


def zero_range_b(p: MutPointer[Byte, ...], count: Int):
    llvm_intrinsic["llvm.memset", NoneType](
        p.unsafe_bitcast[NoneType](), UInt8(0), count, False
    )


def main():
    var buf = stack_allocation[64, Byte]()
    buf[unsafe_offset=0] = 1
    buf[unsafe_offset=63] = 1
    zero_range_a(buf, 64)
    zero_range_b(buf, 64)
    print("zeroed:", buf[unsafe_offset=0] == 0 and buf[unsafe_offset=63] == 0)
