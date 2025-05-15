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

from collections import List
from collections.string import StaticString
from sys import external_call, simdwidthof
from sys.info import _current_target

from algorithm import elementwise
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu.host import DeviceContext
from gpu.host._compile import _get_gpu_target
from gpu.host.info import is_cpu
from memory import memcpy

from utils import IndexList, StaticTuple
from utils.index import product

# ===-----------------------------------------------------------------------===#
# split
# ===-----------------------------------------------------------------------===#


fn split[
    type: DType,
    rank: Int,
    num_outputs: Int,
    target: StaticString,
    trace_description: StaticString,
](
    input: NDBuffer[type, rank],
    axis: Int,
    outputs: StaticTuple[NDBuffer[type, rank, MutableAnyOrigin], num_outputs],
    ctx: DeviceContext,
) raises:
    # check inputs have same rank and same dims except for axis dim
    @parameter
    for i in range(num_outputs):

        @parameter
        for j in range(rank):
            if j != axis and outputs[0].dim(j) != outputs[i].dim(j):
                raise Error(
                    "all split outputs must have the same dimensions in the"
                    " non-split axes"
                )

    var output_sizes = IndexList[num_outputs]()

    @parameter
    for i in range(num_outputs):
        output_sizes[i] = outputs[i].get_shape()[axis]

    @__copy_capture(output_sizes)
    @parameter
    fn elementwise_fn_wrapper[
        width: Int, rank: Int
    ](input_coords: IndexList[rank]) capturing:
        # The associated index in the output tensor
        var output_coords = IndexList[rank]()

        # Which output index to write to
        var output_idx = 0

        # The current shape
        var axis_output_dim = input_coords[axis]

        # First determine which output we should write to
        @parameter
        for i in range(num_outputs):
            if axis_output_dim < output_sizes[i]:
                break
            axis_output_dim -= output_sizes[i]
            output_idx += 1

        # Then derive the output coordinate
        @parameter
        for i in range(rank):
            if i == axis:
                output_coords[i] = axis_output_dim
            else:
                output_coords[i] = input_coords[i]

        var value = input.load[width=width](input_coords)

        # Hack to get around current shortcomings with origins.
        rebind[NDBuffer[type, rank, MutableAnyOrigin]](
            outputs[output_idx]
        ).store(output_coords, value)

    # Can vectorize only if not splitting over last dim.
    if axis != rank - 1:
        alias compile_target = _current_target() if is_cpu[
            target
        ]() else _get_gpu_target()
        alias target_simd_width = simdwidthof[type, target=compile_target]()

        elementwise[
            elementwise_fn_wrapper,
            target_simd_width,
            target=target,
            _trace_description=trace_description,
        ](input.get_shape(), ctx)
    else:
        elementwise[
            elementwise_fn_wrapper,
            1,
            target=target,
            _trace_description=trace_description,
        ](input.get_shape(), ctx)
