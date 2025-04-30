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


# ===-----------------------------------------------------------------------===#
# _argn
# ===-----------------------------------------------------------------------===#

from math import align_down, ceildiv, iota
from sys.info import simdwidthof

from algorithm import sync_parallelize
from algorithm.functional import _get_num_workers
from buffer import NDBuffer
from builtin.math import max as _max
from builtin.math import min as _min

from utils.index import IndexList


fn _argn[
    is_max: Bool
](input: NDBuffer, axis: Int, output: NDBuffer[mut=True, *_]) raises:
    """
    Finds the indices of the maximum/minimum element along the specified axis.

    Parameters:
        is_max: If True compute then compute argmax, otherwise compute the
                argmin.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
    """
    alias rank = input.rank
    alias simd_width = simdwidthof[input.type]()

    var canonical_axis = axis
    if canonical_axis < 0:
        canonical_axis += rank
    if not 0 <= canonical_axis < rank:
        raise Error("axis must be between [0, <input rank>)")

    # TODO: Generalize to mid axis.
    if canonical_axis != rank - 1:
        raise Error("axis other than innermost not supported yet")

    @parameter
    for subaxis in range(rank):
        if subaxis == canonical_axis:
            if output.dim(subaxis) != 1:
                raise Error("expected axis to have size 1 in output")
        elif input.dim(subaxis) != output.dim(subaxis):
            raise Error("input and output dims must match aside from 'axis'")

    var axis_size = input.dim(canonical_axis)
    var input_stride: Int
    var output_stride: Int
    var chunk_size: Int
    var parallel_size = 1

    @parameter
    if rank == 1:
        input_stride = input.num_elements()
        output_stride = output.num_elements()
        chunk_size = 1
    else:
        input_stride = input.stride(canonical_axis - 1)
        output_stride = output.stride(canonical_axis - 1)

        for i in range(canonical_axis):
            parallel_size *= input.dim(i)

        # don't over-schedule if parallel_size < _get_num_workers output
        var num_workers = _min(
            _get_num_workers(input.get_shape().flattened_length()),
            parallel_size,
        )
        chunk_size = ceildiv(parallel_size, num_workers)

    @__copy_capture(
        axis_size, chunk_size, output_stride, input_stride, parallel_size
    )
    @parameter
    fn task_func(task_id: Int):
        @parameter
        @always_inline
        fn cmpeq[
            type: DType, simd_width: Int
        ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
            DType.bool, simd_width
        ]:
            @parameter
            if is_max:
                return a <= b
            else:
                return a >= b

        @parameter
        @always_inline
        fn cmp[
            type: DType, simd_width: Int
        ](a: SIMD[type, simd_width], b: SIMD[type, simd_width]) -> SIMD[
            DType.bool, simd_width
        ]:
            @parameter
            if is_max:
                return a < b
            else:
                return a > b

        # iterate over flattened axes
        var start = task_id * chunk_size
        var end = _min((task_id + 1) * chunk_size, parallel_size)
        for i in range(start, end):
            var input_offset = i * input_stride
            var output_offset = i * output_stride
            var input_dim_ptr = input.data.offset(input_offset)
            var output_dim_ptr = output.data.offset(output_offset)
            var global_val: Scalar[input.type]

            # initialize limits
            @parameter
            if is_max:
                global_val = Scalar[input.type].MIN
            else:
                global_val = Scalar[input.type].MAX

            # initialize vector of maximal/minimal values
            var global_values: SIMD[input.type, simd_width]
            if axis_size < simd_width:
                global_values = global_val
            else:
                global_values = input_dim_ptr.load[width=simd_width]()

            # iterate over values evenly divisible by simd_width
            var indices = iota[output.type, simd_width]()
            var global_indices = indices
            var last_simd_index = align_down(axis_size, simd_width)
            for j in range(simd_width, last_simd_index, simd_width):
                var curr_values = input_dim_ptr.load[width=simd_width](j)
                indices += simd_width

                var mask = cmpeq(curr_values, global_values)
                global_indices = mask.select(global_indices, indices)
                global_values = mask.select(global_values, curr_values)

            @parameter
            if is_max:
                global_val = global_values.reduce_max()
            else:
                global_val = global_values.reduce_min()

            # Check trailing indices.
            var idx = Scalar[output.type](0)
            var found_min: Bool = False
            for j in range(last_simd_index, axis_size, 1):
                var elem = input_dim_ptr.load(j)
                if cmp(global_val, elem):
                    global_val = elem
                    idx = j
                    found_min = True

            # handle the case where min wasn't in trailing values
            if not found_min:
                var matching = global_values == global_val
                var min_indices = matching.select(
                    global_indices, Scalar[output.type].MAX
                )
                idx = min_indices.reduce_min()
            output_dim_ptr[] = idx

    sync_parallelize[task_func](parallel_size)


# ===-----------------------------------------------------------------------===#
# argmax
# ===-----------------------------------------------------------------------===#


fn argmax(
    input: NDBuffer,
    axis: Int,
    output: NDBuffer[mut=True, *_],
) raises:
    """
    Finds the indices of the maximum element along the specified axis.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
    """

    _argn[is_max=True](input, axis, output)


fn argmax(
    input: NDBuffer,
    axis_buf: NDBuffer,
    output: NDBuffer[mut=True, *_],
) raises:
    """
    Finds the indices of the maximum element along the specified axis.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
    """

    argmax(input, Int(axis_buf[0]), output)


# ===-----------------------------------------------------------------------===#
# argmin
# ===-----------------------------------------------------------------------===#


fn argmin(
    input: NDBuffer,
    axis: Int,
    output: NDBuffer[mut=True, *_],
) raises:
    """
    Finds the indices of the minimum element along the specified axis.

    Args:
        input: The input tensor.
        axis: The axis.
        output: The output tensor.
    """

    _argn[is_max=False](input, axis, output)


fn argmin(
    input: NDBuffer,
    axis_buf: NDBuffer,
    output: NDBuffer[mut=True, *_],
) raises:
    """
    Finds the indices of the minimum element along the specified axis.

    Args:
        input: The input tensor.
        axis_buf: The axis tensor.
        output: The axis tensor.
    """

    argmin(input, Int(axis_buf[0]), output)
