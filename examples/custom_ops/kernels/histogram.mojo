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

from math import ceildiv

from gpu import block_dim, block_idx, thread_idx, global_idx
from gpu.host import DeviceContext
from runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice
from algorithm import vectorize, sync_parallelize
from algorithm.functional import _get_num_workers
from memory import memset
from sys import simdwidthof, sizeof
from os import Atomic

from utils.index import IndexList
from gpu.host.info import Info, is_cpu, is_gpu

from memory import UnsafePointer

alias bin_width = Int(UInt8.MAX)


fn _histogram_cpu(out: ManagedTensorSlice, input: ManagedTensorSlice):
    for i in range(input.dim_size(0)):
        out[Int(input[i])] += 1


fn _histogram_gpu(
    output: ManagedTensorSlice,
    input: ManagedTensorSlice,
    ctx_ptr: DeviceContextPtr,
) raises:
    alias block_dim = 1024

    fn kernel(
        output: UnsafePointer[Int64], input: UnsafePointer[UInt8], n: Int
    ):
        var tid = global_idx.x

        if tid >= n:
            return

        _ = Atomic._fetch_add(output + Int(input[tid]), 1)

    var n = input.dim_size(0)

    var grid_dim = ceildiv(n, block_dim)

    var ctx = ctx_ptr.get_device_context()

    ctx.enqueue_function[kernel](
        output.unsafe_ptr(),
        input.unsafe_ptr(),
        n,
        block_dim=block_dim,
        grid_dim=grid_dim,
    )


@compiler.register("histogram", num_dps_outputs=1)
struct Histogram:
    @staticmethod
    fn execute[
        target: StringLiteral
    ](
        out: ManagedTensorSlice[type = DType.int64, rank=1],
        input: ManagedTensorSlice[type = DType.uint8, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        _histogram_cpu(out, input) if is_cpu[target]() else _histogram_gpu(
            out, input, ctx
        )
