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

from sys.intrinsics import strided_load

from gpu import AddressSpace
from gpu.host._compile import _compile_code_asm
from memory import UnsafePointer
from testing import assert_true


fn strided_load_kernel[
    *, type: DType = DType.uint32, width: Int = 1
](
    output: UnsafePointer[SIMD[type, width]],
    ptr: UnsafePointer[Scalar[type], address_space = AddressSpace.GENERIC],
    stride: Int,
):
    output[] = strided_load[width](ptr, stride)


def test_strided_load():
    assert_true(
        "@llvm.masked.gather"
        in _compile_code_asm[
            strided_load_kernel[width=4], emission_kind="llvm"
        ]()
    )


def main():
    test_strided_load()
