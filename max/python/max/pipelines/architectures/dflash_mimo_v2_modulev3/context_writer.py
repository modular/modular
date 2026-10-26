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
"""The DFlash drafter's context writer on ModuleV3: target taps to context K/V.

The drafter sees the target only through K/V: ``ctx = hidden_norm(fc(taps))``,
then per drafter layer ``k = rope(k_norm(k_proj(ctx)))`` and
``v = attention_value_scale * v_proj(ctx)`` at the rows' own positions, with no
``input_layernorm``. Every graph that commits target rows while speculation is
on writes their context through the drafter's
:meth:`~.dflash_mimo_v2.DFlashMiMoV2.write_context`, so the fused spec graph
fills the drafter cache the same way the graph-API build does.

This module holds the tap projection the drafter's layers share: ``fc`` is
row parallel under tensor parallelism (each device multiplies its slice of
the tap features and the mesh sums the partial products ahead of
``hidden_norm``) and a plain ``Linear`` under data parallelism, where each
one-device replica projects its own rows in full. The RoPE table every
layer's K is rotated through lives here too, replicated at every device
count.
"""

from __future__ import annotations

from collections.abc import Sequence

from max.experimental import functional as F
from max.experimental.nn import Linear, Module, RMSNorm
from max.experimental.nn.common_layers.linear import RowParallelLinear
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.rotary_embedding import (
    RotaryEmbedding,
)
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor

from ..dflash_mimo_v2 import DFlashMiMoV2Config

__all__ = ["DFlashContextWriter", "_heads_per_shard"]


def _heads_per_shard(heads: int, num_shards: int, what: str) -> int:
    if heads % num_shards:
        raise ValueError(
            f"DFlash MiMo-V2: {heads} {what} heads do not split over"
            f" {num_shards} devices."
        )
    return heads // num_shards


class DFlashContextWriter(Module[[], None]):
    """The drafter's tap projection: ``fc``, ``hidden_norm`` and the RoPE.

    The K/V the layers write from the projected context are the same ones a
    block query attends to, so both paths go through each layer's attention
    module's ``store_kv``. No ``forward``: the drafter runs its named
    entries (:meth:`~.dflash_mimo_v2.DFlashMiMoV2.write_context`,
    ``forward_block``), never ``__call__``.
    """

    def __init__(self, config: DFlashMiMoV2Config) -> None:
        self.config = config
        # One replica per device under data parallelism: every weight stays
        # whole and each replica projects its own rows, as the target's
        # attention does. Tensor parallelism keeps the row-parallel fc, whose
        # shards' partial sums the mesh reduces.
        self.data_parallel = config.data_parallel_degree > 1
        self.rope = RotaryEmbedding(
            dim=config.rotary_dim * config.num_attention_heads,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            head_dim=config.rotary_dim,
            interleaved=False,
        )
        fc_cls = Linear if self.data_parallel else RowParallelLinear
        self.fc = fc_cls(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _freqs_cis(self, h: Tensor) -> Tensor:
        """The RoPE table, replicated, in its stored float32.

        The drafter rotates K in the table's own dtype, as the graph-API
        drafter does: `fused_qk_ragged_rope` rounds once per rotation, and
        casting the table to the residual stream's bf16 would pre-round
        cos/sin and shift every cached key the later block queries read.
        """
        return self.rope.freqs_cis.to(h.mesh)

    def combine_taps(self, taps: Sequence[Tensor]) -> Tensor:
        """Returns ``hidden_norm(fc(cat(taps)))`` on the mesh.

        Args:
            taps: The target's residual stream after each
                ``target_layer_ids`` layer, in that order, ``[rows,
                hidden]`` -- replicated over tensor parallelism, each
                replica's own rows under data parallelism.
        """
        features = F.concat(list(taps), axis=-1)
        mapping = features.mapping
        if self.data_parallel:
            # The replicated fc projects each replica's rows in full, so the
            # result is complete per shard and no partial sums exist to
            # reduce.
            return self.hidden_norm(self.fc(features).to(mapping))
        # Row-parallel fc, done explicitly: each device slices its share of
        # the concatenated taps and multiplies it against its weight shard
        # (a local slice, not a collective); the mesh then sums the partial
        # products ahead of hidden_norm, as the graph-API writer does.
        sharded = features.to(NamedMapping(features.mesh, (None, TP)))
        return self.hidden_norm(self.fc(sharded).to(mapping))
