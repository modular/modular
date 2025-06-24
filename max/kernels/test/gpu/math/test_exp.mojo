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

from math import exp
from sys import simdwidthof

from algorithm.functional import elementwise
from buffer import NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from testing import *

from utils import Index, IndexList


def run_elementwise[type: DType](ctx: DeviceContext):
    alias length = 256

    alias pack_size = simdwidthof[type, target = get_gpu_target()]()

    var in_device = ctx.enqueue_create_buffer[type](length)
    var out_device = ctx.enqueue_create_buffer[type](length)

    with in_device.map_to_host() as in_host:
        for i in range(length):
            in_host[i] = 0.001 * (Scalar[type](i) - length // 2)

    var in_buffer = NDBuffer[type, 1](in_device._unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[type, 1](out_device._unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)

        out_buffer.store[width=simd_width](
            idx, exp(in_buffer.load[width=simd_width](idx))
        )

    elementwise[func, pack_size, target="gpu"](IndexList[1](length), ctx)

    with in_device.map_to_host() as in_host, out_device.map_to_host() as out_host:
        for i in range(length):
            var msg = String(
                "values did not match at position ",
                i,
                " for dtype=",
                type,
                " and value=",
                in_host[i],
            )

            @parameter
            if type is DType.float32:
                assert_almost_equal(
                    out_host[i],
                    exp(in_host[i]),
                    msg=msg,
                    atol=1e-08,
                    rtol=1e-05,
                )
            else:
                assert_almost_equal(
                    out_host[i],
                    exp(in_host[i]),
                    msg=msg,
                    atol=1e-04,
                    rtol=1e-03,
                )


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.float16](ctx)
        run_elementwise[DType.bfloat16](ctx)
        run_elementwise[DType.float32](ctx)
