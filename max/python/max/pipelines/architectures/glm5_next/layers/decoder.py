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

"""The GLM-5.3-Flash decoder layer.

Because mHC replaces the residual add, this layer -- not the sublayers -- owns
the residual arithmetic and both of the block's layernorms. The published
interface is ``tasks/glm53-scope/contracts/mhc-decoder-layer.md``; read it
before writing a sublayer against this class.

The layer is generic over each sublayer's per-step input bundle so that the KDA,
sparse-MLA and MoE lanes can each declare their own without editing this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar

from max.dtype import DType
from max.graph import ShardingStrategy, TensorValue
from max.nn.layer import Module
from max.nn.norm import RMSNorm
from max.nn.transformer.transformer import forward_sharded_layers

from ..model_config import Glm5NextConfig
from .hyper_connection import HyperConnection, hyper_connection_site

__all__ = ["Glm5NextDecoderLayer", "Glm5NextSublayer"]

SublayerInputsT = TypeVar("SublayerInputsT", contravariant=True)
AttnInputsT = TypeVar("AttnInputsT")
MlpInputsT = TypeVar("MlpInputsT")
_SiteInputsT = TypeVar("_SiteInputsT")


class Glm5NextSublayer(Protocol[SublayerInputsT]):
    """One sublayer of a decoder layer: KDA, sparse MLA, MoE or dense MLP.

    A sublayer takes an already-normalized sequence and returns its output
    alone. It must not add a residual and must not apply an input layernorm --
    :class:`Glm5NextDecoderLayer` owns both, because the mHC write-back is not
    an add and the input layernorm is folded into the mHC mapping.
    """

    def __call__(
        self, xs: list[TensorValue], inputs: SublayerInputsT
    ) -> list[TensorValue]:
        """Runs the sublayer on one normalized sequence per device.

        Args:
            xs: ``[total_tokens, hidden_size]`` per device.
            inputs: The lane's own per-step bundle.

        Returns:
            ``[total_tokens, hidden_size]`` per device, after whatever
            collective the sublayer's own output projection needs.
        """
        ...


class Glm5NextDecoderLayer(Module, Generic[AttnInputsT, MlpInputsT]):
    """A GLM-5.3-Flash decoder layer, with or without hyper-connections.

    With mHC (layers 0-44) the residual carried in and out is
    ``[total_tokens, hc_mult, hidden_size]``, and each of the two sites maps the
    streams to a collapse/placement/mix triple, runs its sublayer on the
    collapsed sequence, then writes the output back into all streams. Without it
    (the MTP draft layer at index 45, whose checkpoint carries no ``hc_*``
    tensors) the residual is ``[total_tokens, hidden_size]`` and the layer is an
    ordinary pre-norm block. Both cases take and return the same argument list,
    so the MTP lane reuses the sublayers unchanged.

    Args:
        config: The model config.
        layer_idx: This layer's index, used only for error messages.
        self_attn: The attention sublayer, KDA or sparse MLA.
        mlp: The feed-forward sublayer, MoE or dense MLP.
        mhc: Whether this layer has hyper-connections. Defaults to
            :attr:`Glm5NextConfig.mhc`; the MTP layer passes ``False``.
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        layer_idx: int,
        *,
        self_attn: Glm5NextSublayer[AttnInputsT],
        mlp: Glm5NextSublayer[MlpInputsT],
        mhc: bool | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.mhc = config.mhc if mhc is None else mhc
        self.self_attn = self_attn
        self.mlp = mlp

        devices = config.devices
        compute_dtype = config.compute_dtype
        self.input_layernorm = _norm(config, compute_dtype)
        self.post_attention_layernorm = _norm(config, compute_dtype)
        replicate = ShardingStrategy.replicate(len(devices))
        self.input_layernorm.sharding_strategy = replicate
        self.post_attention_layernorm.sharding_strategy = replicate
        self.input_layernorm_shards = self.input_layernorm.shard(devices)
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(devices)
        )

        if self.mhc:
            self.attn_hc = hyper_connection_site(config, "hc_attn")
            self.ffn_hc = hyper_connection_site(config, "hc_ffn")
            self.attn_hc.sharding_strategy = replicate
            self.ffn_hc.sharding_strategy = replicate
            self.attn_hc_shards = self.attn_hc.shard(devices)
            self.ffn_hc_shards = self.ffn_hc.shard(devices)

    def __call__(
        self,
        streams: list[TensorValue],
        attn_inputs: AttnInputsT,
        mlp_inputs: MlpInputsT,
    ) -> list[TensorValue]:
        """Runs both sublayers and both residual sites.

        The bundles are positional rather than keyword-only because
        :meth:`~max.nn.layer.Module.build_subgraph` invokes the layer as
        ``self(*inputs)`` when the stack is compiled as subgraphs.

        Args:
            streams: The residual, one tensor per device --
                ``[total_tokens, hc_mult, hidden_size]`` with mHC, and
                ``[total_tokens, hidden_size]`` without.
            attn_inputs: The attention sublayer's per-step bundle.
            mlp_inputs: The feed-forward sublayer's per-step bundle.

        Returns:
            The updated residual, same shapes and devices as ``streams``.
        """
        hs = self._site(
            streams,
            self.self_attn,
            attn_inputs,
            self.input_layernorm_shards,
            self.attn_hc_shards if self.mhc else None,
        )
        return self._site(
            hs,
            self.mlp,
            mlp_inputs,
            self.post_attention_layernorm_shards,
            self.ffn_hc_shards if self.mhc else None,
        )

    def _site(
        self,
        streams: list[TensorValue],
        sublayer: Glm5NextSublayer[_SiteInputsT],
        inputs: _SiteInputsT,
        norm_shards: Sequence[RMSNorm],
        hc_shards: Sequence[HyperConnection] | None,
    ) -> list[TensorValue]:
        """One residual site: read the streams, run the sublayer, write back."""
        if hc_shards is None:
            xs = forward_sharded_layers(norm_shards, streams)
            ys = sublayer(xs, inputs)
            return [x + y for x, y in zip(streams, ys, strict=True)]

        mixings = [
            hc(s, norm.weight)
            for hc, norm, s in zip(hc_shards, norm_shards, streams, strict=True)
        ]
        ys = sublayer([m.xs for m in mixings], inputs)
        return [
            hc.write_back(s, y, m)
            for hc, s, y, m in zip(hc_shards, streams, ys, mixings, strict=True)
        ]


def _norm(config: Glm5NextConfig, dtype: DType) -> RMSNorm:
    return RMSNorm(
        config.hidden_size,
        dtype,
        config.rms_norm_eps,
        multiply_before_cast=False,
    )
