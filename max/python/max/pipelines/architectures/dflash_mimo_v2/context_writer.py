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
"""The DFlash drafter's context writer: target taps to drafter context K/V.

The drafter sees the target only through K/V: ``ctx = hidden_norm(fc(taps))``,
then per drafter layer ``k = rope(k_norm(k_proj(ctx)))`` and
``v = attention_value_scale * v_proj(ctx)`` at the rows' own positions, with no
``input_layernorm``. Every graph that commits target rows while speculation is
on writes their context through this module, so the fused spec graph and the
base graph with the writer attached fill the drafter cache identically.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import ClassVar

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.kernels import (
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer.transformer import forward_sharded_layers
from typing_extensions import Self

from .model_config import DFlashMiMoV2Config


def _heads_per_shard(heads: int, num_shards: int, what: str) -> int:
    if heads % num_shards:
        raise ValueError(
            f"DFlash MiMo-V2: {heads} {what} heads do not split over"
            f" {num_shards} devices."
        )
    return heads // num_shards


def _rms_norm(dim: int, config: DFlashMiMoV2Config) -> RMSNorm:
    # vLLM's RMSNorm, like Qwen3's, casts to the input dtype before the gain.
    return RMSNorm(
        dim, config.dtype, config.rms_norm_eps, multiply_before_cast=False
    )


def _replicated_norm(
    dim: int, config: DFlashMiMoV2Config
) -> tuple[RMSNorm, list[RMSNorm]]:
    norm = _rms_norm(dim, config)
    norm.sharding_strategy = ShardingStrategy.replicate(len(config.devices))
    return norm, list(norm.shard(config.devices))


class DFlashContextAttention(Module, Shardable):
    """A drafter layer's K/V projection, shared by context and block rows.

    The K/V a block query attends to must be computed the same way whether it
    came from a target tap or from the block itself, so both go through
    :meth:`store_kv`.
    """

    def __init__(
        self,
        config: DFlashMiMoV2Config,
        *,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = config.devices[shard_index]
        self.num_key_value_heads = _heads_per_shard(
            config.num_key_value_heads, num_shards, "KV"
        )
        kv_dim = self.num_key_value_heads * config.head_dim
        self.k_proj = Linear(
            config.hidden_size, kv_dim, config.dtype, self.device
        )
        self.v_proj = Linear(
            config.hidden_size, kv_dim, config.dtype, self.device
        )
        self.k_norm = _rms_norm(config.head_dim, config)
        self._sharding_strategy: ShardingStrategy | None = None

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """The head-parallel strategy set by :meth:`shard`'s caller."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "DFlash MiMo-V2 attention shards by head (tensor parallel)."
            )
        self._shard_by_head(strategy.num_devices)
        self._sharding_strategy = strategy

    def _shard_by_head(self, num_devices: int) -> None:
        self.k_proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.v_proj.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        self.k_norm.sharding_strategy = ShardingStrategy.replicate(num_devices)

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Self]:
        """Splits the heads over ``devices``, one shard per device."""
        devices = list(devices)
        k_proj = self.k_proj.shard(devices)
        v_proj = self.v_proj.shard(devices)
        k_norm = self.k_norm.shard(devices)
        shards = []
        for i in range(len(devices)):
            shard = type(self)(
                self.config, shard_index=i, num_shards=len(devices)
            )
            shard.k_proj, shard.v_proj, shard.k_norm = (
                k_proj[i],
                v_proj[i],
                k_norm[i],
            )
            shards.append(shard)
        return shards

    def store_kv(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        q: TensorValue | None = None,
    ) -> TensorValue:
        """Writes the K/V of ``x``'s rows at their positions and ropes ``q``.

        The only kernel that rotates a NeoX prefix of the head rotates the
        cached K together with a query, so without ``q`` the K rows stand in
        for it and the result is discarded.

        Args:
            layer_idx: The drafter layer.
            x: ``[rows, hidden]`` inputs, ragged by ``input_row_offsets``.
            kv_collection: The drafter cache, whose lengths place the rows.
            freqs_cis: The ``[positions, rotary_dim]`` RoPE table.
            input_row_offsets: Ragged offsets of ``x``.
            q: ``[rows, heads, head_dim]`` queries to rope, if any.

        Returns:
            ``q``, or the K rows without it, roped at the rows' positions.
        """
        config = self.config
        head_dim = config.head_dim
        kv_shape = (-1, self.num_key_value_heads, head_dim)
        k = self.k_norm(self.k_proj(x).reshape((-1, head_dim))).reshape(
            kv_shape
        )
        v = self.v_proj(x)
        # A bfloat16 constant would round the scale itself (0.612 -> 0.6133).
        v = ops.cast(
            ops.cast(v, DType.float32) * config.attention_value_scale, v.dtype
        ).reshape(kv_shape)
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, v, input_row_offsets, layer_idx)
        return fused_qk_ragged_rope(
            config.kv_params,
            k if q is None else q,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=False,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Writes the K/V of ``x``'s rows; see :meth:`store_kv`."""
        return self.store_kv(
            layer_idx, x, kv_collection, freqs_cis, input_row_offsets
        )


class DFlashContextLayer(Module):
    """One drafter layer as the context writer sees it: its K/V projection.

    Calling a layer writes its context K/V.
    """

    attention_cls: ClassVar[type[DFlashContextAttention]] = (
        DFlashContextAttention
    )

    def __init__(self, config: DFlashMiMoV2Config) -> None:
        super().__init__()
        self.self_attn = self.attention_cls(config)
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(config.devices)
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

    def __call__(
        self,
        layer_idx: TensorValue,
        ctx: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> None:
        """Writes this layer's context K/V on every device."""
        for attn, x, kv, freqs, offsets in zip(
            self.self_attn_shards,
            ctx,
            kv_collections,
            freqs_cis,
            input_row_offsets,
            strict=True,
        ):
            attn.store_kv(layer_idx, x, kv, freqs, offsets)


class DFlashContextWriter(Module):
    """Writes drafter context K/V for committed target rows.

    Built alone, it declares only the weights the writer reads (``fc``,
    ``hidden_norm`` and each layer's ``k_proj``, ``v_proj`` and ``k_norm``),
    under the drafter checkpoint's names.

    ``fc`` is row parallel: each device multiplies its slice of the tap
    features, and one allreduce sums the partial products ahead of
    ``hidden_norm``.
    """

    def __init__(
        self,
        config: DFlashMiMoV2Config,
        layers: Sequence[DFlashContextLayer] | None = None,
    ) -> None:
        """Builds the writer.

        Args:
            config: The drafter configuration.
            layers: The drafter's layers, when the writer is part of the
                drafter. ``None`` builds K/V-only layers.
        """
        super().__init__()
        self.config = config
        devices = config.devices
        self.rope = RotaryEmbedding(
            dim=config.rotary_dim * config.num_attention_heads,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.rotary_dim,
            interleaved=False,
        )
        self.fc = Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            config.dtype,
            devices[0],
        )
        self.fc.sharding_strategy = ShardingStrategy.columnwise(len(devices))
        self.fc_shards = self.fc.shard(devices)
        self.hidden_norm, self.hidden_norm_shards = _replicated_norm(
            config.hidden_size, config
        )
        self.layers = LayerList(
            list(layers)
            if layers is not None
            else [
                DFlashContextLayer(config)
                for _ in range(config.num_hidden_layers)
            ]
        )

    def _freqs_cis(self) -> list[TensorValue]:
        return [
            self.rope.freqs_cis.to(device) for device in self.config.devices
        ]

    def combine_taps(
        self,
        taps: Sequence[Sequence[TensorValue]],
        signal_buffers: Sequence[BufferValue] | None = None,
    ) -> list[TensorValue]:
        """Returns ``fc(cat(taps))`` on every device, before ``hidden_norm``.

        Args:
            taps: Per device, the ``[rows, hidden]`` residual stream after each
                of ``target_layer_ids``, in that order, replicated.
            signal_buffers: The allreduce buffers; needed on more than one
                device.

        Returns:
            The context projection, ``[rows, hidden]`` per device.
        """
        num_taps = len(self.config.target_layer_ids)
        partials = []
        start = 0
        for device_taps, fc in zip(taps, self.fc_shards, strict=True):
            if len(device_taps) != num_taps:
                raise ValueError(
                    f"DFlash MiMo-V2: expected {num_taps} taps per device, got"
                    f" {len(device_taps)}."
                )
            width = int(fc.weight.shape[1])
            features = ops.concat(device_taps, axis=-1)[
                :, start : start + width
            ]
            partials.append(fc(features))
            start += width
        if len(partials) == 1:
            return partials
        assert signal_buffers is not None, (
            "row-parallel fc needs signal buffers"
        )
        return ops.allreduce.sum(partials, signal_buffers)

    def write(
        self,
        ctx: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
    ) -> None:
        """Writes every layer's context K/V from :meth:`combine_taps`'s output.

        Args:
            ctx: Per device, the context projection.
            input_row_offsets: Per device, ragged offsets of the rows.
            kv_collections: Per device, the drafter cache, whose lengths are
                the rows' first positions.
        """
        hidden = forward_sharded_layers(self.hidden_norm_shards, list(ctx))
        freqs_cis = self._freqs_cis()
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, DFlashContextLayer)
            layer(
                ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU()),
                hidden,
                kv_collections,
                freqs_cis,
                input_row_offsets,
            )

    def __call__(
        self,
        taps: Sequence[Sequence[TensorValue]],
        input_row_offsets: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
        signal_buffers: Sequence[BufferValue] | None = None,
    ) -> None:
        """Writes the context K/V of the taps' rows into the drafter cache.

        Args:
            taps: See :meth:`combine_taps`.
            input_row_offsets: See :meth:`write`.
            kv_collections: See :meth:`write`.
            signal_buffers: See :meth:`combine_taps`.
        """
        self.write(
            self.combine_taps(taps, signal_buffers),
            input_row_offsets,
            kv_collections,
        )
