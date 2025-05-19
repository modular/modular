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

from gpu.host._compile import _compile_code_asm, _get_gpu_target
from gpu.intrinsics import ldg
from layout import Layout, LayoutTensor
from memory import UnsafePointer
from testing import assert_true


fn ldg_kernel(i8: UnsafePointer[Int8]):
    i8.store(1, ldg(i8))


fn layout_kernel(a: LayoutTensor[mut=False, DType.int8, Layout.row_major(1)]):
    a[1] = a[0]


def test_ldg_kernel[emission_kind: StaticString]() -> String:
    return _compile_code_asm[
        ldg_kernel,
        emission_kind=emission_kind,
        target = _get_gpu_target["sm_90a"](),
    ]()


def test_ldg_kernel():
    var llvm = test_ldg_kernel["llvm"]()
    assert_true("!invariant.load !1" in llvm)
    var asm = test_ldg_kernel["asm"]()
    assert_true("ld.global.nc.u8" in asm)


def test_layout_kernel[emission_kind: StaticString]() -> String:
    return _compile_code_asm[
        layout_kernel,
        emission_kind=emission_kind,
        target = _get_gpu_target["sm_90a"](),
    ]()


def test_layout_kernel():
    var llvm = test_layout_kernel["llvm"]()
    assert_true("!invariant.load !1" in llvm)
    var asm = test_layout_kernel["asm"]()
    assert_true("ld.global.nc.u8" in asm)


def main():
    test_ldg_kernel()
    test_layout_kernel()
