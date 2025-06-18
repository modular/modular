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
"""Multi-GPU allgather implementation that gathers values from multiple GPUs
into an output buffer.
"""

from collections import InlineArray

from buffer import NDBuffer
from gpu.host import DeviceBuffer, DeviceContext

from utils import IndexList


@always_inline
fn allgather[
    type: DType,
    rank: Int,
    ngpus: Int, //,
](
    input_buffers: InlineArray[NDBuffer[type, rank, MutableAnyOrigin], ngpus],
    output_buffers: InlineArray[
        NDBuffer[type, rank, MutableAnyOrigin], ngpus * ngpus
    ],
    ctxs: List[DeviceContext],
) raises:
    """
    Performs all-gather across GPUs with variadic output.

    Each device receives individual copies of all input buffers.

    Parameters:
        type: DType - The data type of tensor elements.
        rank: Int - Number of dimensions in input tensors.
        ngpus: Int - Number of GPUs participating in all-gather.

    Args:
        input_buffers: Input buffers from each GPU.
        output_buffers: Flat array of ngpus * ngpus output buffers.
                       Layout: output_buffers[device_idx * ngpus + input_idx]
                       contains device_idx's copy of input_idx's data.
        ctxs: List of device contexts for participating GPUs.
    """

    var device_buffers = List[DeviceBuffer[type]](capacity=ngpus)

    # Assemble input buffers from all devices.
    for device_idx in range(ngpus):
        device_buffers.append(
            DeviceBuffer(
                ctxs[device_idx],
                input_buffers[device_idx].data,
                input_buffers[device_idx].num_elements(),
                owning=False,
            )
        )

    for device_idx in range(ngpus):
        var curr_ctx = ctxs[device_idx]

        # Copy each input to this device as a separate buffer.
        for input_idx in range(ngpus):
            # Calculate flat index for this output buffer.
            var output_idx = device_idx * ngpus + input_idx

            var output_device_buffer = DeviceBuffer(
                curr_ctx,
                output_buffers[output_idx].data,
                output_buffers[output_idx].num_elements(),
                owning=False,
            )

            # Copy from input device to current device.
            curr_ctx.enqueue_copy(
                output_device_buffer,
                device_buffers[input_idx],
            )
