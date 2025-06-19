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

from math import log, log2, log10
from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from testing import assert_almost_equal

from utils import Index, IndexList


def run_elementwise[
    dtype: DType, log_fn: fn (x: SIMD) -> __type_of(x)
](ctx: DeviceContext):
    alias length = 8192

    alias pack_size = simdwidthof[dtype, target = get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[dtype](length)
    var out_device = ctx.enqueue_create_buffer[dtype](length)

    alias epsilon = 0.001
    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = Scalar[dtype](i) + epsilon

    var in_buffer = NDBuffer[dtype, 1](in_device._unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[dtype, 1](out_device._unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)
        var val = in_buffer.load[width=simd_width](idx)
        var result = log_fn(val)
        out_buffer.store[width=simd_width](idx, result)

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var expected_value = log_fn(in_host[i])

            alias atol = 1e-07 if dtype is DType.float32 else 1e-4
            alias rtol = 2e-07 if dtype is DType.float32 else 2e-2
            assert_almost_equal(
                out_host[i],
                expected_value,
                msg=String("values did not match at position ", i),
                atol=atol,
                rtol=rtol,
            )


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.float32, log](ctx)
        run_elementwise[DType.float32, log10](ctx)
        run_elementwise[DType.float32, log2](ctx)
        run_elementwise[DType.float16, log](ctx)
        run_elementwise[DType.float16, log10](ctx)
        run_elementwise[DType.float16, log2](ctx)
        run_elementwise[DType.bfloat16, log](ctx)
        run_elementwise[DType.bfloat16, log10](ctx)
        run_elementwise[DType.bfloat16, log2](ctx)
