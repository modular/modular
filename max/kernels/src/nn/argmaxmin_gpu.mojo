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


from buffer import NDBuffer
from gpu.host import DeviceContext
from nn.topk import topk_gpu


fn argmaxmin_gpu[
    type: DType, output_type: DType, rank: Int, largest: Bool
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    output: NDBuffer[mut=True, output_type, rank],
) raises:
    """
    Wraps the Top-K GPU kernel with K=1 to perform argmax on the inner-most
    dimension.

    Parameters:
        type: DType - The data type of the input tensor.
        output_type: DType - The data type of the output tensor.
        rank: Int - The rank of the input tensor.
        largest: Bool - Whether to perform argmax or argmin.
    Args:
        ctx: DeviceContext - The device context.
        input: NDBuffer[type, rank] - The input tensor allocated on the device.
        output: NDBuffer[type, rank] - The output tensor allocated on the device.
    """
    constrained[rank > 0, "Input rank must be positive"]()
    alias K = 1

    var out_vals_shape = input.get_shape()
    out_vals_shape[rank - 1] = K
    var out_vals_buf = ctx.enqueue_create_buffer[type](
        out_vals_shape.flattened_length()
    )
    var out_vals = NDBuffer[type, rank](
        out_vals_buf._unsafe_ptr(), out_vals_shape
    )

    topk_gpu[sampling=False, largest=largest](
        ctx,
        K,
        input,
        out_vals,
        output,
    )

    _ = out_vals_buf^


fn argmax_gpu[
    type: DType, output_type: DType, rank: Int
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    output: NDBuffer[mut=True, output_type, rank],
) raises:
    argmaxmin_gpu[largest=True](ctx, input, output)


fn argmin_gpu[
    type: DType, output_type: DType, rank: Int
](
    ctx: DeviceContext,
    input: NDBuffer[type, rank],
    output: NDBuffer[mut=True, output_type, rank],
) raises:
    argmaxmin_gpu[largest=False](ctx, input, output)
