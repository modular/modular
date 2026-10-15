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
"""Per-step inputs to a GLM-5.3-Flash KDA sublayer.

The decoder layer is generic over each sublayer's bundle, so the KDA bundle
lives here rather than in a shared dataclass nobody owns. Core builds one per
KDA layer per step; only the row ids differ between layers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from max import tree
from max.graph import BufferValue, TensorValue
from max.nn.kv_cache import RecurrentStateInputsPerDevice

from ..state_cache import CONV_LEAF_ID, RECURRENT_LEAF_ID, leaf_inputs

__all__ = [
    "KdaReplayInputs",
    "KdaSublayerInputs",
    "kda_sublayer_inputs",
]


@tree.dataclass(frozen=True, kw_only=True)
class KdaSublayerInputs:
    """One KDA layer's per-step inputs, one entry per device.

    Each pool spans every layer's rows, and the kernels mutate it in place at
    the rows the ids name, so nothing here is a graph output.
    """

    signal_buffers: list[BufferValue]
    """Allreduce signal buffers for ``o_proj``'s partial sums."""

    input_row_offsets: list[TensorValue]
    """``[batch_size + 1]`` exclusive prefix offsets over the packed tokens.

    Cast as each kernel needs it -- the conv op takes uint32 and the
    recurrence ops take int32 -- so either integer dtype is accepted here.
    """

    conv_pools: list[BufferValue]
    """``[num_rows, conv_dim, conv_kernel_size - 1]``, every layer's rows."""

    conv_row_ids: list[TensorValue]
    """``[batch_size]`` conv row this layer runs each sequence in."""

    recurrent_pools: list[BufferValue]
    """``[num_rows, num_heads, head_dim, head_dim]``, every layer's rows."""

    recurrent_row_ids: list[TensorValue]
    """``[batch_size]`` recurrent row this layer runs each sequence in.

    Its own leaf's rows, not the conv leaf's: the two are separate pool leaves
    and a block of one names nothing in the other.
    """


class KdaReplayInputs(NamedTuple):
    """One KDA layer's per-token inputs to the two state kernels.

    Speculative decoding runs the verify pass over K+1 positions and then has
    to land the pools on the accepted prefix instead. Every op feeding these
    tensors is causal or pointwise, so re-running the kernels over the accepted
    rows from the pre-verify state reproduces the state the verify pass held at
    that length -- the argument Qwen3.5's
    ``unified_mtp_qwen3_5/state_rollback.py`` rests on.

    What does not carry over from Qwen3.5 is the shape of ``raw_gate``: KDA's
    forget gate is one value per *channel* per head, not one scalar per head,
    so a rollback that replays a ``[total_tokens, num_heads]`` decay is
    replaying the wrong tensor.
    """

    qkv: TensorValue
    """``[total_tokens, conv_dim]`` conv input, pre-convolution."""

    conv_weight: TensorValue
    """``[conv_dim, conv_kernel_size]`` depthwise weights."""

    raw_gate: TensorValue
    """``[total_tokens, num_heads, head_dim]`` forget-gate pre-activation.

    Per channel. Pre-``dt_bias`` and pre-activation, matching what the
    recurrence op consumes.
    """

    beta_logits: TensorValue
    """``[total_tokens, num_heads]`` input-gate logits, pre-sigmoid."""


def kda_sublayer_inputs(
    *,
    kda_layers: Sequence[int],
    state: Sequence[RecurrentStateInputsPerDevice[TensorValue, BufferValue]],
    signal_buffers: list[BufferValue],
    input_row_offsets: list[TensorValue],
) -> dict[int, KdaSublayerInputs]:
    """Splits the recurrent-state graph inputs into one bundle per KDA layer.

    Every layer shares one pool per leaf per device and differs only in the
    rows it runs in: the ids arrive folded ``[batch_size, num_kda_layers]``,
    and KDA layer ``l`` reads column ``l``. That column index is the one thing
    about this wiring that is wrong silently -- reading the *decoder* index
    instead of the KDA position binds layer 0's state to layer 3, which is
    neither a shape error nor a crash, just a worse model. So it is done here,
    once.

    Args:
        kda_layers: Decoder indices of the KDA layers, in schedule order --
            :attr:`Glm5NextConfig.kda_layers`. Position in this sequence is
            the column a layer reads.
        state: The recurrent-state inputs, one entry per device.
        signal_buffers: One per device, shared by every layer.
        input_row_offsets: One per device, shared by every layer.

    Returns:
        One bundle per entry of ``kda_layers``, keyed by decoder index.

    Raises:
        ValueError: If a per-device list does not have one entry per device.
    """
    num_devices = len(state)
    for name, per_device in (
        ("signal_buffers", signal_buffers),
        ("input_row_offsets", input_row_offsets),
    ):
        if len(per_device) != num_devices:
            raise ValueError(
                f"{name} must have one entry per device, got "
                f"{len(per_device)} for {num_devices} devices."
            )
    conv = [leaf_inputs(device, CONV_LEAF_ID) for device in state]
    recurrent = [leaf_inputs(device, RECURRENT_LEAF_ID) for device in state]
    return {
        layer_idx: KdaSublayerInputs(
            signal_buffers=signal_buffers,
            input_row_offsets=input_row_offsets,
            conv_pools=[leaf.pool for leaf in conv],
            conv_row_ids=[leaf.live_row_id(position) for leaf in conv],
            recurrent_pools=[leaf.pool for leaf in recurrent],
            recurrent_row_ids=[
                leaf.live_row_id(position) for leaf in recurrent
            ],
        )
        for position, layer_idx in enumerate(kda_layers)
    }
