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

from math import iota
from sys import alignof, sizeof, num_physical_cores
from algorithm import parallelize_over_rows
from bit import log2_floor
from compiler import register
from gpu import WARP_SIZE, barrier, warp
from gpu.memory import AddressSpace, external_memory
from max.tensor import ManagedTensorSlice
from memory import Span
from utils.index import IndexList
from utils.numerics import min_or_neg_inf


@value
@register_passable("trivial")
struct TopKElement[T: DType]:
    """Stores the value with it's index."""

    var idx: Int32
    var val: Scalar[T]

    fn __gt__(self, rhs: Self) -> Bool:
        return self.val > rhs.val


@register("top_k_custom", num_dps_outputs=2)
struct TopK:
    """Registers the `top_k_custom` op, allowing python to use it from the `max`
    package. This is a simplified version without bottom_k and sorting options,
    or fused sampling. The purpose is to demonstrate concisely how you can
    implement your own custom ops in Mojo that can be called from Python. MAX
    has the "mo.top_k" op which is feature complete.
    """

    @staticmethod
    fn execute[
        type: DType,
        rank: Int,
        //,  # Forces the previous two params to be inferred from the args
        K: Int,
        target: StringLiteral,
    ](
        out_vals: ManagedTensorSlice[type=type, rank=rank],
        out_idxs: ManagedTensorSlice[type = DType.int32, rank=rank],
        in_vals: ManagedTensorSlice[type=type, rank=rank],
        ctx: DeviceContextPtr,
    ) raises:
        constrained[rank == 2, "rank must be 2"]()
        constrained[
            not (target == "gpu" and K > WARP_SIZE),
            "K can't be larger than warp size",
        ]()

        var shape = in_vals.shape()
        var batch_size = shape[0]
        var dev_ctx = ctx.get_device_context()

        @parameter
        fn top_k_gpu[
            K: Int,
        ](
            out_vals: __type_of(out_vals),
            out_idxs: __type_of(out_idxs),
            in_vals: __type_of(in_vals),
        ):
            var bid = block_idx.x
            var tid = thread_idx.x

            # Get a pointer to shared memory for the indices and values
            var top_k_sram = external_memory[
                TopKElement[type],
                address_space = AddressSpace.SHARED,
                alignment = alignof[TopKElement[type]](),
            ]()

            # Threads put their corresponding index and value into shared memory
            top_k_sram[tid] = TopKElement(tid, in_vals[bid, tid])
            # Finish packing the values across threads in this block
            barrier()

            @parameter
            for i in range(K):
                var reduced = top_k_sram[tid]
                alias limit = log2_floor(WARP_SIZE)

                # TODO(KERN-1544): `gpu.shuffle.warp_max` support index/value
                @parameter
                for j in reversed(range(limit)):
                    alias offset = 1 << j
                    # Parallel reduction using warp shuffle. Each thread gets a
                    # value from a thread 'offset' positions higher, keeping the
                    # larger value.
                    var shuffled = TopKElement(
                        warp.shuffle_down(reduced.idx, offset),
                        warp.shuffle_down(reduced.val, offset),
                    )
                    reduced = max(reduced, shuffled)

                # Wait for all threads to finish reducing their values
                barrier()

                # Thread 0 now has the reduced max value for this index
                if tid == 0:
                    # Store the reduced top_k index and value in global memory
                    out_vals[bid, i] = reduced.val
                    out_idxs[bid, i] = reduced.idx

                    # Remove found maximum from consideration in the next iter
                    var index = reduced.idx % block_dim.x
                    top_k_sram[index].val = min_or_neg_inf[type]()

        @parameter
        if target == "gpu":
            dev_ctx.enqueue_function[top_k_gpu[K]](
                out_vals,
                out_idxs,
                in_vals,
                grid_dim=batch_size,  # One block per batch
                block_dim=K,  # One thread per K
                shared_mem_bytes=K * sizeof[TopKElement[type]](),
            )
        else:

            @parameter
            fn top_k_cpu(start_idx: Int, end_idx: Int):
                for row_idx in range(start_idx, end_idx):
                    var offset = (row_idx * K)
                    iota(out_idxs.unsafe_ptr() + offset, K)

                    @parameter
                    fn val_greater_than(lhs: Int32, rhs: Int32) -> Bool:
                        return (
                            in_vals[row_idx, Int(lhs)]
                            > in_vals[row_idx, Int(rhs)]
                        )

                    sort[val_greater_than](
                        Span(out_idxs.unsafe_ptr() + offset, K)
                    )

                    for i in range(K):
                        var sorted_idx = Int(out_idxs[row_idx, i])
                        out_vals[row_idx, i] = in_vals[row_idx, sorted_idx]

            parallelize_over_rows[top_k_cpu](shape, axis=1, grain_size=1)
