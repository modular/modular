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

from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from testing import assert_almost_equal

from utils import Index, IndexList

alias type = DType.float32


def run_elementwise(exponent: BFloat16, ctx: DeviceContext):
    alias length = 256

    alias pack_size = simdwidthof[type, target = get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[type](length)
    var out_device = ctx.enqueue_create_buffer[type](length)

    # Add a small constant to avoid 0^-pow.
    alias epsilon = 0.001
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = abs((Scalar[type](i) - length // 2) + epsilon)

    var in_buffer = NDBuffer[type, 1](in_device._unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[type, 1](out_device._unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer, exponent)
    @parameter
    fn func[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)

        var val = in_buffer.load[width=simd_width](idx).cast[DType.bfloat16]()
        var result = val ** SIMD[DType.bfloat16, simd_width](exponent)
        out_buffer.store[width=simd_width](idx, result.cast[DType.float32]())

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = in_host[i] ** exponent.cast[DType.float32]()
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=1e-04,
                rtol=2e-02,
            )


def main():
    # NOTE: This is expected to fail. Keeping this around as a negative test
    # so we know when its fixed.
    with DeviceContext() as ctx:
        run_elementwise(0.375, ctx)
