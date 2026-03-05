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
"""Op implementation for transfer_to."""

from __future__ import annotations

from max.mlir.dialects import mo, rmo

from ..graph import Graph
from ..type import Device, DeviceRef, TensorType, _ChainType
from ..value import StrongTensorValueLike, TensorValue


def transfer_to(
    x: StrongTensorValueLike, device: Device | DeviceRef
) -> TensorValue:
    """Graph execution-time device transfer operation.

    Inserts a transfer node into the compiled graph that moves ``x`` to
    ``device`` at execution time. This is a **graph-level operation**: it
    operates on symbolic :obj:`TensorValue` objects during graph tracing and
    is baked into the compiled graph as an ``mo.transfer`` MLIR op.

    This is distinct from :meth:`~max.experimental.nn.Module.to`, which is a
    **pre-compilation** operation that moves stored weight tensors on the
    Python host before the graph is built. Use ``transfer_to`` when you need
    to route an activation tensor between devices inside :meth:`forward`
    (e.g., host-to-device input staging, device-to-host output retrieval, or
    cross-GPU tensor movement for multi-device models).

    Implementation notes:

    - Host↔device transfers (CPU↔GPU) use the graph's immutable root chain
      so they can be hoisted to model initialization by the optimizer.
    - Device-to-device transfers (GPU↔GPU) join both per-device chains to
      prevent reordering that would deadlock multi-device collectives.
    - If source and destination device are identical, this is a no-op.

    Args:
        x: The input tensor to transfer.
        device: The target device.

    Returns:
        A new :obj:`TensorValue` on the specified device.
    """
    x = TensorValue(x)
    device = DeviceRef.from_device(device)

    if device == x.type.device:
        return x

    graph = Graph.current

    # Host-to-device and device-to-host transfers are globally safe and should
    # not be chained to per-device execution.
    # Use the graph's immutable root chain so these transfers can be hoisted to
    # model init.
    if x.type.device.is_cpu() or device.is_cpu():
        in_chain = graph.always_ready_chain
        # NOTE: leave the out chain unused: transfers between device and host
        # don't pose a deadlock risk with other multi-device ops, so their
        # sequence doesn't matter.
        (result, _) = graph._add_op(
            rmo.mo_transfer,
            TensorType(dtype=x.dtype, shape=x.shape, device=device).to_mlir(),
            _ChainType().to_mlir(),
            x,
            in_chain,
        )
        return result.tensor

    #   2 device execution flow
    #   -------------------------------------------------
    #   GPU:0 chain                         GPU:1 chain
    #   -----------                         -----------
    #   ...                                 ...
    #   collective(0,1)                     collective(0,1)
    #        |                                    |
    #        +----------- both launched ----------+   <-- safe point
    #                     in_chain = mo.chain.create([ch0, ch1])
    #                     mo.transfer[in_chain] %x : gpu:0 -> gpu:1
    #
    # Only schedule the D2D transfer when both per-device chains guarantee that
    # the collective op has launched on each device.
    # This prevents reordering that could otherwise deadlock multi-device
    # collectives and transfers.
    in_chain = graph._add_op(
        mo.chain_create,
        [graph.device_chains[x.type.device], graph.device_chains[device]],
    )[0]

    (result, out_chain) = graph._add_op(
        rmo.mo_transfer,
        TensorType(dtype=x.dtype, shape=x.shape, device=device).to_mlir(),
        _ChainType().to_mlir(),
        x,
        in_chain,
    )

    # Set source and destination device chains to ensure ordering of subsequent
    # multi-device ops as well.
    graph.device_chains[device] = out_chain
    graph.device_chains[x.type.device] = out_chain

    return result.tensor
