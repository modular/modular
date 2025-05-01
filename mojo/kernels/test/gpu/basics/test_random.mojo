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
# RUN: %mojo-no-debug %s

from math import isqrt, sqrt
from sys import has_neon, simdwidthof

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.random import Random
from testing import *

from utils.index import Index


def run_elementwise[type: DType](ctx: DeviceContext):
    alias length = 256

    alias pack_size = simdwidthof[type, target = _get_gpu_target()]()

    var in_host = NDBuffer[
        type, 1, MutableAnyOrigin, DimList(length)
    ].stack_allocation()
    var out_host = NDBuffer[
        type, 1, MutableAnyOrigin, DimList(length)
    ].stack_allocation()

    var flattened_length = in_host.num_elements()
    for i in range(length):
        in_host[i] = 0.001 * abs(Scalar[type](i) - length // 2)

    var in_device = ctx.enqueue_create_buffer[type](flattened_length)
    var out_device = ctx.enqueue_create_buffer[type](flattened_length)

    ctx.enqueue_copy(in_device, in_host.data)

    var in_buffer = NDBuffer[type, 1](in_device._unsafe_ptr(), Index(length))
    var out_buffer = NDBuffer[type, 1](out_device._unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer, in_buffer)
    @parameter
    fn func[simd_width: Int, rank: Int](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)

        var rng_state = Random(seed=idx0[0])
        var rng = rng_state.step_uniform()

        @parameter
        if simd_width == 1:
            out_buffer[idx] = rng[0].cast[type]()
        else:

            @parameter
            for i in range(simd_width):
                out_buffer[idx + i] = rng[i % len(rng)].cast[type]()

    elementwise[func, 4, target="gpu"](Index(length), ctx)

    ctx.enqueue_copy(out_host.data, out_device)

    ctx.synchronize()

    for i in range(length):
        print(out_host[i])

    _ = in_device
    _ = out_device


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.float16](ctx)
        run_elementwise[DType.float32](ctx)
        run_elementwise[DType.float64](ctx)
