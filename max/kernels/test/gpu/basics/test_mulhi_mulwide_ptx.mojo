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

# `mulhi` and `mulwide` are written as a widening multiply, and rely on the
# backend to select a single PTX instruction for it. The operands are kernel
# arguments so that nothing folds away.

from std.math import mulhi, mulwide
from std.testing import assert_equal, assert_true

from max.gpu.host.compile import _compile_code, get_gpu_target

comptime _SM80 = get_gpu_target["sm_80"]()


def mulhi_u32(out_p: MutPointer[UInt32, MutAnyOrigin], a: UInt32, b: UInt32):
    out_p[0] = mulhi(a, b)


def mulhi_s32(out_p: MutPointer[Int32, MutAnyOrigin], a: Int32, b: Int32):
    out_p[0] = mulhi(a, b)


def mulhi_u64(out_p: MutPointer[UInt64, MutAnyOrigin], a: UInt64, b: UInt64):
    out_p[0] = mulhi(a, b)


def mulhi_s64(out_p: MutPointer[Int64, MutAnyOrigin], a: Int64, b: Int64):
    out_p[0] = mulhi(a, b)


def mulwide_u32(out_p: MutPointer[UInt64, MutAnyOrigin], a: UInt32, b: UInt32):
    out_p[0] = mulwide(a, b)


def mulwide_s32(out_p: MutPointer[Int64, MutAnyOrigin], a: Int32, b: Int32):
    out_p[0] = mulwide(a, b)


def _assert_single_multiply(asm: String, instruction: StaticString) raises:
    assert_true(instruction in asm, msg=instruction)
    # A widened multiply that was not combined expands into several
    # `mul.lo`/`mul.hi`/`mad` instructions.
    assert_equal(asm.count("mul."), 1, msg=instruction)
    assert_true("mad." not in asm, msg=instruction)


def main() raises:
    _assert_single_multiply(
        _compile_code[mulhi_u32, target=_SM80]().asm, "mul.hi.u32"
    )
    _assert_single_multiply(
        _compile_code[mulhi_s32, target=_SM80]().asm, "mul.hi.s32"
    )
    _assert_single_multiply(
        _compile_code[mulhi_u64, target=_SM80]().asm, "mul.hi.u64"
    )
    _assert_single_multiply(
        _compile_code[mulhi_s64, target=_SM80]().asm, "mul.hi.s64"
    )
    _assert_single_multiply(
        _compile_code[mulwide_u32, target=_SM80]().asm, "mul.wide.u32"
    )
    _assert_single_multiply(
        _compile_code[mulwide_s32, target=_SM80]().asm, "mul.wide.s32"
    )
