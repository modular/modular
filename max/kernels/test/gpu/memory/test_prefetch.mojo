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

from std.memory import stack_allocation
from std.sys.intrinsics import prefetch

from max.gpu.host import get_gpu_target
from max.gpu.host.compile import _compile_code
from std.testing import assert_false, assert_true

comptime _SM100A = get_gpu_target["sm_100a"]()


def do_prefetch[
    dtype: DType, *, offset: Int = 0
](addr: ImmPointer[Scalar[dtype], ImmutAnyOrigin]):
    prefetch(addr.unsafe_offset(offset))


def do_prefetch_global(
    addr: Pointer[Float32, ImmutAnyOrigin, address_space=.GLOBAL]
):
    prefetch(addr)


def do_prefetch_constant(
    addr: Pointer[Float32, ImmutAnyOrigin, address_space=.CONSTANT]
):
    prefetch(addr)


def do_prefetch_stack[
    address_space: AddressSpace
](output: MutPointer[Float32, MutAnyOrigin]):
    var buffer = stack_allocation[4, Float32, address_space=address_space]()
    buffer[0] = output[0]
    prefetch(buffer)
    output[1] = buffer[0]


def test_prefetch_nvidia_generic() raises:
    assert_true(
        "prefetch.L2 ["
        in _compile_code[do_prefetch[.float16], target=_SM100A]()
    )
    assert_true(
        "prefetch.L2 ["
        in _compile_code[do_prefetch[.float32], target=_SM100A]()
    )
    assert_true(
        "prefetch.L2 [" in _compile_code[do_prefetch[.int32], target=_SM100A]()
    )
    assert_true(
        "prefetch.L2 ["
        in _compile_code[do_prefetch[.int64, offset=42], target=_SM100A]()
    )


def test_prefetch_nvidia_address_spaces() raises:
    assert_true(
        "prefetch.global.L2 ["
        in _compile_code[do_prefetch_global, target=_SM100A]()
    )
    assert_true(
        "prefetch.local.L2 ["
        in _compile_code[do_prefetch_stack[.LOCAL], target=_SM100A]()
    )
    # PTX has no `prefetch` for shared or constant memory, so the hint is
    # dropped. Match `.L2 [` rather than `prefetch.`, which debug-info builds
    # also emit in the `.file` directive naming this test.
    assert_false(
        ".L2 [" in _compile_code[do_prefetch_stack[.SHARED], target=_SM100A]()
    )
    assert_false(
        ".L2 [" in _compile_code[do_prefetch_constant, target=_SM100A]()
    )


def main() raises:
    test_prefetch_nvidia_generic()
    test_prefetch_nvidia_address_spaces()
