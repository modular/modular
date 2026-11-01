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
"""Provides a CPU implementation of the argmax reduction."""

from std.math import align_down, ceildiv, iota
from std.sys import align_of
from std.sys.info import simd_width_of

from max.algorithm import sync_parallelize
from max.algorithm.functional import _get_num_workers
from max.gpu.host import DeviceContext
from std.math.math import min as _min
from layout import Coord, Idx, TileTensor
from layout.tile_layout import Layout


def argmax(
    input: TileTensor[mut=False, ...],
    axis: Int,
    output: TileTensor[mut=True, ...],
    ctx: Optional[DeviceContext.CPU] = None,
) raises:
    """
    Finds the indices of the maximum element along the specified axis.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
        ctx: The context to execute the work on.
    """
    comptime rank = input.rank
    comptime simd_width = simd_width_of[input.dtype]()

    var canonical_axis = axis
    if canonical_axis < 0:
        canonical_axis += rank
    if not 0 <= canonical_axis < rank:
        raise Error("axis must be between [0, <input rank>)")

    # TODO: Generalize to mid axis.
    if canonical_axis != rank - 1:
        raise Error("axis other than innermost not supported yet")

    comptime for subaxis in range(rank):
        var output_subaxis = output.dim(subaxis)
        var input_subaxis = output.dim(subaxis)
        if subaxis == canonical_axis:
            if output_subaxis != 1:
                raise Error("expected axis to have size 1 in output")
        elif input_subaxis != output_subaxis:
            raise Error("input and output dims must match aside from 'axis'")

    var axis_size = Int(input.dim(canonical_axis))
    var input_stride: Int
    var output_stride: Int
    var chunk_size: Int
    var parallel_size = 1

    comptime if rank == 1:
        input_stride = input.num_elements()
        output_stride = output.num_elements()
        chunk_size = 1
    else:
        input_stride = Int(input.dynamic_stride(canonical_axis - 1))
        output_stride = Int(output.dynamic_stride(canonical_axis - 1))

        for i in range(canonical_axis):
            parallel_size *= Int(input.dim(i))

        # don't over-schedule if parallel_size < _get_num_workers output
        var num_workers = _min(
            _get_num_workers(input.num_elements(), ctx=ctx),
            parallel_size,
        )
        chunk_size = ceildiv(parallel_size, num_workers)

    def task_func(
        task_id: Int,
    ) {
        var axis_size,
        var chunk_size,
        var output_stride,
        var input_stride,
        var parallel_size,
        imm,
    }:
        @inline(.always)
        def cmpeq[
            dtype: DType, simd_width: SIMDLength
        ](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
            DType.bool, simd_width
        ]:
            return a.le(b)

        @inline(.always)
        def cmp[
            dtype: DType, simd_width: SIMDLength
        ](a: SIMD[dtype, simd_width], b: SIMD[dtype, simd_width]) -> SIMD[
            DType.bool, simd_width
        ]:
            return a.lt(b)

        # iterate over flattened axes
        var start = task_id * chunk_size
        var end = _min((task_id + 1) * chunk_size, parallel_size)
        var input_rows = input.reshape(
            Layout(Coord(parallel_size, axis_size), Coord(input_stride, Idx[1]))
        )
        var output_rows = output.reshape(
            Layout(Coord(parallel_size), Coord(output_stride))
        )
        for i in range(start, end):
            var global_val = Scalar[input.dtype].MIN

            # initialize vector of maximal values
            var global_values: SIMD[input.dtype, simd_width]
            if axis_size < simd_width:
                global_values = global_val
            else:
                global_values = input_rows.load[
                    width=simd_width, alignment=align_of[input.dtype]()
                ]((i, 0))

            # iterate over values evenly divisible by simd_width
            var indices = iota[output.dtype, simd_width]()
            var global_indices = indices
            var last_simd_index = align_down(axis_size, simd_width)
            for j in range(simd_width, last_simd_index, simd_width):
                var curr_values = input_rows.load[
                    width=simd_width, alignment=align_of[input.dtype]()
                ]((i, j))
                indices += Scalar[output.dtype](simd_width)

                var mask = cmpeq(curr_values, global_values)
                global_indices = mask.select(global_indices, indices)
                global_values = mask.select(global_values, curr_values)

            global_val = global_values.reduce_max()

            # Check trailing indices.
            var idx = Scalar[output.dtype](0)
            var found_min: Bool = False
            for j in range(last_simd_index, axis_size, 1):
                var elem = input_rows.load[width=1]((i, j))
                if cmp(global_val, elem):
                    global_val = elem
                    idx = Scalar[output.dtype](j)
                    found_min = True

            # handle the case where min wasn't in trailing values
            if not found_min:
                var matching = global_values.eq(global_val)
                var min_indices = matching.select(
                    global_indices, Scalar[output.dtype].MAX
                )
                idx = min_indices.reduce_min()
            output_rows.store((i,), idx)

    sync_parallelize(task_func, parallel_size, ctx)
