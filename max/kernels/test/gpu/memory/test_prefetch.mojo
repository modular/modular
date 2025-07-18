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

from sys.intrinsics import prefetch

from gpu.host.compile import _compile_code
from gpu.host import get_gpu_target
from testing import assert_true


fn do_prefetch[
    dtype: DType, *, offset: Int = 0
](addr: UnsafePointer[Scalar[dtype]]):
    prefetch(addr + offset)


def test_prefetch_mi300x():
    assert_true(
        "llvm.prefetch "
        in _compile_code[
            do_prefetch[DType.float16], target = get_gpu_target["mi300x"]()
        ]()
    )
    assert_true(
        "llvm.prefetch "
        in _compile_code[
            do_prefetch[DType.float32], target = get_gpu_target["mi300x"]()
        ]()
    )
    assert_true(
        "llvm.prefetch "
        in _compile_code[
            do_prefetch[DType.int32], target = get_gpu_target["mi300x"]()
        ]()
    )

    assert_true(
        "llvm.prefetch "
        in _compile_code[
            do_prefetch[DType.int64, offset=42],
            target = get_gpu_target["mi300x"](),
        ]()
    )


def test_prefetch_nvidia():
    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[DType.float16], target = get_gpu_target["sm_80"]()
        ]()
    )
    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[DType.float32], target = get_gpu_target["sm_80"]()
        ]()
    )
    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[DType.int32], target = get_gpu_target["sm_80"]()
        ]()
    )

    assert_true(
        "prefetch.global.L2 "
        in _compile_code[
            do_prefetch[DType.int64, offset=42],
            target = get_gpu_target["sm_80"](),
        ]()
    )


def main():
    test_prefetch_nvidia()
