# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from gpu import thread_idx
from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from gpu.intrinsics import ldg
from testing import *


fn register_intrinsics(
    i8: UnsafePointer[Int8],
    ui8: UnsafePointer[UInt8],
    i16: UnsafePointer[Int16],
    ui16: UnsafePointer[UInt16],
    i32: UnsafePointer[Int32],
    ui32: UnsafePointer[UInt32],
    i64: UnsafePointer[Int64],
    ui64: UnsafePointer[UInt64],
    f32: UnsafePointer[Float32],
    f64: UnsafePointer[Float64],
):
    # Note we perform the store purely to avoid the compiler from optimizing
    # away the statements.
    var tid = thread_idx.x
    i8.store(tid, ldg(i8))
    ui8.store(tid, ldg(ui8))
    i16.store(tid, ldg(i16))
    ui16.store(tid, ldg(ui16))
    i32.store(tid, ldg(i32))
    ui32.store(tid, ldg(ui32))
    i64.store(tid, ldg(i64))
    ui64.store(tid, ldg(ui64))
    f32.store(tid, ldg(f32))
    f64.store(tid, ldg(f64))


@always_inline
fn _verify_register_intrinsics(asm: StringSlice) raises -> None:
    assert_true("ld.global.nc.b8" in asm)
    assert_true("ld.global.nc.b16" in asm)
    assert_true("ld.global.nc.b32" in asm)
    assert_true("ld.global.nc.b64" in asm)
    assert_true("ld.global.nc.b32" in asm)
    assert_true("ld.global.nc.b64" in asm)


def test_register_intrinsics_sm80():
    var asm = _compile_code[
        register_intrinsics, target = get_gpu_target["sm_80"]()
    ]().asm
    _verify_register_intrinsics(asm)


def test_register_intrinsics_sm90():
    var asm = _compile_code[
        register_intrinsics,
        target = get_gpu_target["sm_90"](),
    ]().asm
    _verify_register_intrinsics(asm)


def main():
    test_register_intrinsics_sm80()
    test_register_intrinsics_sm90()
