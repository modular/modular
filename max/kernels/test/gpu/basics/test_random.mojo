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

from sys import simd_width_of

from algorithm.functional import elementwise
from buffer import DimList, NDBuffer
from gpu import *
from gpu.host import DeviceContext, get_gpu_target
from random import NormalRandom, Random
from testing import *
from sys import has_apple_gpu_accelerator

from utils.index import Index, IndexList


def run_elementwise[
    dtype: DType, distribution: String = "uniform"
](ctx: DeviceContext):
    comptime length = 256

    comptime pack_size = simd_width_of[dtype, target = get_gpu_target()]()

    var out_host = NDBuffer[
        dtype, 1, MutAnyOrigin, DimList(length)
    ].stack_allocation()

    var out_device = ctx.enqueue_create_buffer[dtype](length)
    var out_buffer = NDBuffer[dtype, 1](out_device.unsafe_ptr(), Index(length))

    @always_inline
    @__copy_capture(out_buffer)
    @parameter
    fn func_uniform[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)

        var rng_state = Random(seed=UInt64(idx0[0]))
        var rng = rng_state.step_uniform()

        comptime if simd_width == 1:
            out_buffer[idx] = rng[0].cast[dtype]()
        else:
            comptime for i in range(simd_width):
                out_buffer[idx + IndexList[1](i)] = rng[i % len(rng)].cast[
                    dtype
                ]()

    @always_inline
    @__copy_capture(out_buffer)
    @parameter
    fn func_normal[
        simd_width: Int, rank: Int, alignment: Int = 1
    ](idx0: IndexList[rank]):
        var idx = rebind[IndexList[1]](idx0)
        var rng_state = NormalRandom(seed=UInt64(idx0[0]))
        var rng = rng_state.step_normal()

        comptime if simd_width == 1:
            out_buffer[idx] = rng[0].cast[dtype]()
        else:
            comptime for i in range(simd_width):
                out_buffer[idx + IndexList[1](i)] = rng[i % len(rng)].cast[
                    dtype
                ]()

    comptime if distribution == "uniform":
        elementwise[func_uniform, 4, target="gpu"](Index(length), ctx)
    else:
        elementwise[func_normal, 4, target="gpu"](Index(length), ctx)

    ctx.enqueue_copy(out_host.data, out_device)
    ctx.synchronize()

    print("Testing", distribution, "distribution:")
    for i in range(length):
        print(out_host[i])

    _ = out_device


def main():
    with DeviceContext() as ctx:
        run_elementwise[DType.float16](ctx)
        run_elementwise[DType.float32](ctx)
        run_elementwise[DType.float16, "normal"](ctx)
        run_elementwise[DType.float32, "normal"](ctx)
        if not has_apple_gpu_accelerator():
            # Metal does not support DType.float64
            run_elementwise[DType.float64](ctx)
            run_elementwise[DType.float64, "normal"](ctx)
