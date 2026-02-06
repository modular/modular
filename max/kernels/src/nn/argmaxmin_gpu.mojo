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


from gpu.host import DeviceContext
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.topk import topk_gpu


fn argmaxmin_gpu[
    dtype: DType, output_type: DType, largest: Bool
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_type, ...],
) raises:
    """
    Wraps the Top-K GPU kernel with K=1 to perform argmax on the inner-most
    dimension.

    Parameters:
        dtype: DType - The data dtype of the input tensor.
        output_type: DType - The data dtype of the output tensor.
        largest: Bool - Whether to perform argmax or argmin.
    Args:
        ctx: DeviceContext - The device context.
        input: TileTensor[dtype] - The input tensor allocated on the device.
        output: TileTensor[dtype] - The output tensor allocated on the device.
    """
    comptime assert input.rank > 0, "Input rank must be positive"
    comptime assert (
        input.rank == output.rank
    ), "Input and output rank must be the same"
    comptime K = 1

    var out_vals_shape = coord_to_index_list(input.layout.shape_coord())
    out_vals_shape[input.rank - 1] = K
    var out_vals_buf = ctx.enqueue_create_buffer[dtype](
        out_vals_shape.flattened_length()
    )
    var out_vals = TileTensor(
        out_vals_buf.unsafe_ptr(),
        row_major(Coord(out_vals_shape)),
    )

    topk_gpu[sampling=False, largest=largest](ctx, K, input, out_vals, output)

    _ = out_vals_buf^


fn argmax_gpu[
    dtype: DType, output_type: DType
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_type, ...],
) raises:
    argmaxmin_gpu[largest=True](ctx, input, output)


fn argmin_gpu[
    dtype: DType, output_type: DType
](
    ctx: DeviceContext,
    input: TileTensor[dtype, ...],
    output: TileTensor[mut=True, output_type, ...],
) raises:
    argmaxmin_gpu[largest=False](ctx, input, output)
