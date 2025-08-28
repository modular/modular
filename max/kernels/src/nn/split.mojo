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

from collections.string import StaticString
from sys import simd_width_of
from sys.info import _current_target

from algorithm import elementwise
from gpu.host import DeviceContext
from gpu.host import get_gpu_target
from gpu.host.info import is_cpu
from layout import LayoutTensor, Layout, RuntimeTuple, UNKNOWN_VALUE
from layout.int_tuple import fill_like

from utils import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
# split
# ===-----------------------------------------------------------------------===#


fn split[
    type: DType,
    num_outputs: Int,
    target: StaticString,
    trace_description: StaticString,
    outputs_origin: MutableOrigin,
    outputs_layout: Layout,
](
    input: LayoutTensor[type, **_],
    axis: Int,
    outputs: StaticTuple[
        LayoutTensor[type, outputs_layout, outputs_origin], num_outputs
    ],
    ctx: DeviceContext,
) raises:
    constrained[
        input.rank == outputs[0].rank,
        "Input and outputs must have the same rank.",
    ]()

    # check inputs have same rank and same dims except for axis dim
    @parameter
    for i in range(num_outputs):

        @parameter
        for j in range(input.rank):
            if j != axis and outputs[0].dim[j]() != outputs[i].dim[j]():
                raise Error(
                    "all split outputs must have the same dimensions in the"
                    " non-split axes"
                )

    var output_sizes = IndexList[num_outputs]()

    @parameter
    for i in range(num_outputs):
        output_sizes[i] = outputs[i].dim(axis)

    @__copy_capture(output_sizes)
    @parameter
    fn elementwise_fn_wrapper[
        width: Int, rank: Int, alignment: Int = 1
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

        var idx = input.runtime_layout(
            RuntimeTuple[fill_like(input.layout.shape, UNKNOWN_VALUE)](
                input_coords
            )
        )

        var value = input.ptr.load[width=width](idx)

        var output_ptr_idx = outputs[output_idx].runtime_layout(
            RuntimeTuple[
                fill_like(outputs[output_idx].layout.shape, UNKNOWN_VALUE)
            ](output_coords)
        )

        outputs[output_idx].ptr.store(output_ptr_idx, value)

    # Can vectorize only if not splitting over last dim.
    if axis != input.rank - 1:
        alias compile_target = _current_target() if is_cpu[
            target
        ]() else get_gpu_target()
        alias target_simd_width = simd_width_of[type, target=compile_target]()

        elementwise[
            elementwise_fn_wrapper,
            target_simd_width,
            target=target,
            _trace_description=trace_description,
        ](input.runtime_layout.shape.value, ctx)
    else:
        elementwise[
            elementwise_fn_wrapper,
            1,
            target=target,
            _trace_description=trace_description,
        ](input.runtime_layout.shape.value, ctx)
