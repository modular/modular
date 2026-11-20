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

import extensibility

from max.gpu.host import DeviceContext
from std.math import iota
from std.sys import align_of, size_of

from max.algorithm import parallelize_over_rows
from std.bit import log2_floor
from max.gpu import (
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.primitives import warp
from max.gpu.memory import external_memory
from std.collections import Span

from extensibility import Input, Output, Tensor, TileTensorable
from layout import TensorLayout, TileTensor, coord_to_index_list

from std.utils.coord import Coord

from std.utils.numerics import min_or_neg_inf


@fieldwise_init
struct TopKElement[T: DType](Comparable, TrivialRegisterPassable):
    """Stores a value, its index, and whether it remains eligible."""

    var idx: Int32
    var val: Scalar[Self.T]
    var active: UInt32

    def __eq__(self, rhs: Self) -> Bool:
        return self.active == rhs.active and self.val == rhs.val

    def __lt__(self, rhs: Self) -> Bool:
        if self.active != rhs.active:
            return self.active < rhs.active
        return self.val < rhs.val


@extensibility.register("top_k_custom")
struct TopK:
    """Registers the `top_k_custom` op, allowing python to use it from the `max`
    package. This is a simplified version without bottom_k and sorting options,
    or fused sampling. The purpose is to demonstrate concisely how you can
    implement your own custom ops in Mojo that can be called from Python. MAX
    has the "mo.top_k" op which is feature complete.
    """

    @staticmethod
    def execute[
        K: Int,
    ](
        out_vals: Some[Output & Tensor],
        out_idxs: Some[Output & Tensor],
        in_vals: Some[Input & Tensor],
        ctx: DeviceContext,
    ) raises:
        comptime assert conforms_to(type_of(out_vals), TileTensorable)
        comptime assert conforms_to(type_of(out_idxs), TileTensorable)
        comptime assert conforms_to(type_of(in_vals), TileTensorable)
        comptime dtype = type_of(in_vals).dtype
        comptime assert type_of(out_vals).rank == 2, "rank must be 2"
        comptime assert type_of(out_idxs).rank == 2, "rank must be 2"
        comptime assert type_of(in_vals).rank == 2, "rank must be 2"

        # `run` takes plain pointer-backed views, so rebuild each argument's
        # view from its pointer and layout.
        Self.run[K](
            TileTensor(
                ptr=out_vals.to_tile_tensor[dtype]().unsafe_ptr(),
                layout=out_vals.layout(),
            ),
            TileTensor(
                ptr=out_idxs.to_tile_tensor[.int32]().unsafe_ptr(),
                layout=out_idxs.layout(),
            ),
            TileTensor(
                ptr=in_vals.to_tile_tensor[dtype]().unsafe_ptr(),
                layout=in_vals.layout(),
            ),
            ctx,
        )

    @staticmethod
    def run[
        dtype: DType,
        vals_layout: TensorLayout,
        idxs_layout: TensorLayout,
        in_layout: TensorLayout,
        //,
        K: Int,
    ](
        out_vals: TileTensor[mut=True, dtype, vals_layout, _],
        out_idxs: TileTensor[mut=True, .int32, idxs_layout, _],
        in_vals: TileTensor[dtype, in_layout, _],
        ctx: DeviceContext,
    ) raises:
        """Runs top-k on `TileTensor` views of rank-2 tensors."""
        comptime assert not (
            ctx.target.is_gpu() and K > WARP_SIZE
        ), "K can't be larger than warp size"

        var shape = coord_to_index_list(in_vals.layout.shape_coord())
        var batch_size = shape[0]
        var dev_ctx = ctx

        comptime OutVals = type_of(out_vals)
        comptime OutIdxs = type_of(out_idxs)
        comptime InVals = type_of(in_vals)

        def top_k_gpu[
            K: Int,
        ](out_vals: OutVals, out_idxs: OutIdxs, in_vals: InVals,):
            comptime assert out_vals.flat_rank == 2
            comptime assert out_idxs.flat_rank == 2
            comptime assert in_vals.flat_rank == 2
            var bid = block_idx.x
            var tid = thread_idx.x

            # Get a pointer to shared memory for the indices and values
            var top_k_sram = external_memory[
                TopKElement[dtype],
                address_space=.SHARED,
                alignment=align_of[TopKElement[dtype]](),
            ]()

            # Every lane participates in the full-warp shuffle, but only K
            # lanes contain candidates. Eligibility is separate from the value
            # so legitimate negative infinity remains selectable.
            var value = min_or_neg_inf[dtype]()
            var active = UInt32(0)
            if tid < K:
                value = in_vals[bid, tid][0]
                active = 1
            top_k_sram[unsafe_offset=tid] = TopKElement(
                Int32(tid), value, active
            )
            # Finish packing the values across threads in this block
            barrier()

            comptime for i in range(K):
                var reduced = top_k_sram[unsafe_offset=tid]
                comptime limit = log2_floor(WARP_SIZE)

                # TODO(KERN-1544): `gpu.shuffle.warp_max` support index/value
                comptime for j in reversed(range(limit)):
                    comptime offset = 1 << j
                    # Parallel reduction using warp shuffle. Each thread gets a
                    # value from a thread 'offset' positions higher, keeping the
                    # larger value.
                    var shuffled = TopKElement(
                        warp.shuffle_down(reduced.idx, UInt32(offset)),
                        warp.shuffle_down(reduced.val, UInt32(offset)),
                        warp.shuffle_down(reduced.active, UInt32(offset)),
                    )
                    if tid + offset < WARP_SIZE:
                        reduced = max(reduced, shuffled)

                # Wait for all threads to finish reducing their values
                barrier()

                # Thread 0 now has the reduced max value for this index
                if tid == 0:
                    # Store the reduced top_k index and value in global memory
                    out_vals[bid, i] = reduced.val
                    out_idxs[bid, i] = reduced.idx

                    # Remove found maximum from consideration in the next iter
                    top_k_sram[unsafe_offset=reduced.idx].active = 0

                # Retirement must be visible before the next candidate reload.
                barrier()

        comptime if ctx.target.is_gpu():
            dev_ctx.enqueue_function[top_k_gpu[K]](
                out_vals,
                out_idxs,
                in_vals,
                grid_dim=batch_size,  # One block per batch
                block_dim=WARP_SIZE,
                shared_mem_bytes=WARP_SIZE * size_of[TopKElement[dtype]](),
            )
        else:

            def top_k_cpu(start_idx: Int, end_idx: Int) {imm}:
                for row_idx in range(start_idx, end_idx):
                    var offset = row_idx * K
                    iota(out_idxs.unsafe_ptr().unsafe_offset(offset), K)

                    def val_greater_than(
                        lhs: Int32, rhs: Int32
                    ) {in_vals, row_idx} -> Bool:
                        return in_vals.load[width=1](
                            Coord(row_idx, Int(lhs))
                        ) > in_vals.load[width=1](Coord(row_idx, Int(rhs)))

                    sort(
                        Span(
                            unsafe_ptr=out_idxs.unsafe_ptr().unsafe_offset(
                                offset
                            ),
                            length=K,
                        ),
                        val_greater_than,
                    )

                    for i in range(K):
                        var sorted_idx = Int(
                            out_idxs.load[width=1](Coord(row_idx, i))
                        )
                        out_vals.store[width=1](
                            Coord(row_idx, i),
                            in_vals.load[width=1](Coord(row_idx, sorted_idx)),
                        )

            parallelize_over_rows(top_k_cpu, shape, axis=1, grain_size=1)
