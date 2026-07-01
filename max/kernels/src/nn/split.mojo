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

from std.collections.string import StaticString
from std.sys import simd_width_of
from std.sys.info import _current_target

from std.algorithm import elementwise
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.host.info import is_cpu
from layout import (
    Coord,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
)

from std.utils import IndexList, StaticTuple

# ===-----------------------------------------------------------------------===#
# split
# ===-----------------------------------------------------------------------===#


def split[
    OutputLayoutType: TensorLayout,
    //,
    dtype: DType,
    num_outputs: Int,
    target: StaticString,
    trace_description: StaticString,
    output_origin: MutOrigin,
    *,
    axis: Int,
](
    input: TileTensor[dtype, ...],
    outputs: StaticTuple[
        TileTensor[dtype, OutputLayoutType, output_origin],
        num_outputs,
    ],
    ctx: DeviceContext,
) raises:
    comptime assert (
        input.rank == OutputLayoutType.rank
    ), "Input and outputs must have the same rank."

    # check inputs have same rank and same dims except for axis dim
    comptime for i in range(num_outputs):
        comptime for j in range(input.rank):
            if j != axis and outputs[0].dim[j]() != outputs[i].dim[j]():
                raise Error(
                    "all split outputs must have the same dimensions in the"
                    " non-split axes"
                )

    var output_sizes = IndexList[num_outputs]()

    comptime for i in range(num_outputs):
        output_sizes[i] = Int(outputs[i].dim(axis))

    @always_inline
    def elementwise_fn_wrapper[
        width: Int, alignment: Int = 1
    ](input_coords: Coord) {var output_sizes, var input, var outputs,}:
        # The associated index in the output tensor
        var output_coords = IndexList[input_coords.rank]()
        var input_idx = coord_to_index_list(input_coords)

        # Which output index to write to
        var output_idx = 0

        # The current shape
        var axis_output_dim = input_idx[axis]

        # First determine which output we should write to
        comptime for i in range(num_outputs):
            if axis_output_dim < output_sizes[i]:
                break
            axis_output_dim -= output_sizes[i]
            output_idx += 1

        # Then derive the output coordinate
        comptime for i in range(input_coords.rank):
            if i == axis:
                output_coords[i] = axis_output_dim
            else:
                output_coords[i] = input_idx[i]

        var idx = input.layout(input_coords)

        var value = input.raw_load[width=width](idx)

        # Write through a COMPILE-TIME index into the `outputs` StaticTuple.
        # On Metal (Apple M5), a StaticTuple[TileTensor, N] aggregate indexed
        # at runtime inside a device closure fails to marshal its embedded
        # device pointers -- the kernel reads/writes a host-side pointer and
        # the store lands nowhere, so every output comes back all-zeros (even
        # for N == 2). A compile-time index into the aggregate marshals
        # correctly. Dispatch the runtime `output_idx` to a comptime `i` and
        # store via `outputs[i]` (compile-time). See KB pattern
        # `gpu-kernel-closures-must-copy-capture-tensors` and the runtime-vs-
        # comptime StaticTuple probes (.derived/repro_*statictuple*).
        comptime for i in range(num_outputs):
            if output_idx == i:
                var output_ptr_idx = outputs[i].layout(Coord(output_coords))
                outputs[i].raw_store(output_ptr_idx, value)

    # Can vectorize only if not splitting over last dim.
    if axis != input.rank - 1:
        comptime compile_target = _current_target() if is_cpu[
            target
        ]() else get_gpu_target()
        comptime target_simd_width = simd_width_of[
            dtype, target=compile_target
        ]()

        elementwise[
            simd_width=target_simd_width,
            target=target,
            _trace_description=trace_description,
        ](elementwise_fn_wrapper, input.layout.shape_coord(), ctx)
    else:
        elementwise[
            simd_width=1,
            target=target,
            _trace_description=trace_description,
        ](elementwise_fn_wrapper, input.layout.shape_coord(), ctx)
