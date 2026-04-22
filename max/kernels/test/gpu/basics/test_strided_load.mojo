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

from std.sys.intrinsics import strided_load

from std.gpu.host.compile import _compile_code
from std.testing import assert_true


def strided_load_kernel[
    *, dtype: DType = DType.uint32, width: Int = 1
](
    output: UnsafePointer[SIMD[dtype, width], MutAnyOrigin],
    ptr: UnsafePointer[
        Scalar[dtype], ImmutAnyOrigin, address_space=AddressSpace.GENERIC
    ],
    stride: Int,
):
    output[] = strided_load[width](ptr, stride)


def test_strided_load() raises:
    assert_true(
        "@llvm.masked.gather"
        in _compile_code[strided_load_kernel[width=4], emission_kind="llvm"]()
    )


def main() raises:
    test_strided_load()
