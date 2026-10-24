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
"""Laguna attention layer.

GQA attention, causal or sliding-window depending on the layer's type in the
HF config's ``layer_types``, with two Laguna-specific touches vs the
MiniMax-M2 donor:

1. **Per-head Q/K RMSNorm.** ``q_norm`` and ``k_norm`` gammas are
   ``[head_dim]`` (128), applied per-head after reshape. (Donor was
   ``[num_heads * head_dim]``, applied across all heads at once.)

2. **Softplus output gate** (`g_proj`). ``gate =
   softplus(g_proj(hidden_states))`` scales the post-attention tensor before
   ``o_proj`` -- one scalar per query head under ``gating_mode="per-head"``,
   one per ``(head, head_dim)`` channel under ``"per-element"``.

3. **Partial-rotary lane permutation.** On a layer whose ``rotary_dim`` is
   below ``head_dim``, q and k lanes are reordered before the fused RoPE store
   so the kernel's rotation pairs land on the reference's. See
   :meth:`LagunaAttention._permute_rotary_lanes`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import RotaryEmbedding

# The lane permutation halves both the rotated and pass-through blocks,
# and each half must hold whole (cos, sin) pairs.
_ROTARY_DIM_MULTIPLE = 4


def _softplus(x: TensorValue) -> TensorValue:
    """Computes ``log(1 + exp(x))`` without overflowing on large inputs.

    MAX has no ``ops.softplus``, and the direct form saturates to ``inf`` once
    ``exp(x)`` does, near ``x = 88`` in float32. Folding out ``max(x, 0)``
    leaves a non-positive exponent argument, so the identity below is finite
    across the whole range.
    """
    return ops.max(x, 0.0) + ops.log1p(ops.exp(-ops.abs(x)))


class LagunaAttention(Module, Shardable):
    """Per-layer attention with per-head QK-norm and softplus output gate.

    A sliding layer (``is_sliding``) masks to ``local_window_size`` keys and
    reads the sliding cache leaf; a full layer is plain causal over the full
    cache. ``kv_params`` is the leaf this layer addresses, not the whole tree.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        is_sliding: bool = False,
        local_window_size: int = -1,
        gating_mode: str = "per-element",
        rotary_dim: int | None = None,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float,
        qk_norm_eps: float = 1e-6,
        norm_dtype: DType,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.local_window_size = local_window_size
        if is_sliding and local_window_size <= 0:
            raise ValueError(
                "A sliding-attention Laguna layer needs a positive window, "
                f"got local_window_size={local_window_size}."
            )
        if gating_mode not in ("per-head", "per-element"):
            raise ValueError(
                "Laguna attention gating must be 'per-head' or 'per-element', "
                f"got {gating_mode!r}."
            )
        self.gating_mode = gating_mode
        self.kv_params = kv_params
        # ``None`` means full rotary, which needs no lane permutation.
        self.rotary_dim = (
            kv_params.head_dim if rotary_dim is None else rotary_dim
        )
        if not 0 < self.rotary_dim <= kv_params.head_dim:
            raise ValueError(
                f"Laguna rotary_dim must be in (0, head_dim="
                f"{kv_params.head_dim}], got {self.rotary_dim}."
            )
        if self.rotary_dim % _ROTARY_DIM_MULTIPLE:
            # The permutation halves the rotated block and the pass-through
            # block; an odd half would straddle a (cos, sin) pair.
            raise ValueError(
                f"Laguna rotary_dim must be a multiple of {_ROTARY_DIM_MULTIPLE}, got "
                f"{self.rotary_dim}."
            )
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.norm_dtype = norm_dtype
        self.quant_config = quant_config
        self.linear_cls = linear_cls
        self.scale = scale
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        head_dim = self.kv_params.head_dim
        q_weight_dim = head_dim * num_attention_heads
        kv_weight_dim = head_dim * num_key_value_heads

        # Per-head Q/K RMSNorm gammas: shape [head_dim] (NOT [n_heads*head_dim]).
        # HF reference: q_norm = RMSNorm(head_dim); applied to reshaped
        # [*, n_heads, head_dim] tensors so the same gamma broadcasts over
        # all heads. Same for k_norm over kv heads.
        self.q_norm = Weight(
            "q_norm.weight",
            shape=[head_dim],
            dtype=self.norm_dtype,
            device=devices[0],
        )
        self.k_norm = Weight(
            "k_norm.weight",
            shape=[head_dim],
            dtype=self.norm_dtype,
            device=devices[0],
        )

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        # The checkpoint's ``g_proj.weight`` shape forces the gating mode, so
        # it arrives as a constructor argument rather than being inferred here.
        self.g_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=(
                num_attention_heads
                if gating_mode == "per-head"
                else q_weight_dim
            ),
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "LagunaAttention only supports replicate sharding "
                "(DP+EP mode). TP is not supported."
            )

        self.q_proj.sharding_strategy = strategy
        self.k_proj.sharding_strategy = strategy
        self.v_proj.sharding_strategy = strategy
        self.o_proj.sharding_strategy = strategy
        self.g_proj.sharding_strategy = strategy
        self.q_norm.sharding_strategy = strategy
        self.k_norm.sharding_strategy = strategy

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[LagunaAttention]:
        if not self._sharding_strategy:
            raise ValueError(
                "LagunaAttention layer cannot be sharded because no "
                "sharding strategy was provided."
            )

        devices_list = list(devices)

        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)
        g_proj_shards = self.g_proj.shard(devices_list)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)

        shards: list[LagunaAttention] = []

        for shard_idx, device in enumerate(devices_list):
            # Replicate-only (enforced by the sharding_strategy setter): each
            # shard is a full copy with all heads.
            sharded = LagunaAttention(
                rope=self.rope,
                num_attention_heads=self.n_heads,
                num_key_value_heads=self.num_key_value_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                is_sliding=self.is_sliding,
                local_window_size=self.local_window_size,
                gating_mode=self.gating_mode,
                rotary_dim=self.rotary_dim,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                qk_norm_eps=self.qk_norm_eps,
                norm_dtype=self.norm_dtype,
                quant_config=self.quant_config,
            )

            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.g_proj = g_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]

            shards.append(sharded)

        return shards

    def _per_head_rms_norm(
        self, x: TensorValue, weight: Weight, num_heads: int, head_dim: int
    ) -> TensorValue:
        """Applies per-head RMSNorm to a flat ``[*, num_heads * head_dim]`` tensor.

        The gamma is ``[head_dim]`` and broadcasts over the head axis, matching
        HF ``q_norm``/``k_norm`` applied to ``[*, num_heads, head_dim]``.
        """
        normed = ops.rms_norm(
            x.reshape((-1, num_heads, head_dim)),
            weight=weight.to(x.device),
            epsilon=self.qk_norm_eps,
        )
        return normed.reshape((-1, num_heads * head_dim))

    def _permute_rotary_lanes(
        self, x: TensorValue, num_heads: int, head_dim: int
    ) -> TensorValue:
        """Reorders head lanes so the kernel's RoPE pairs match the reference's.

        The reference rotates lanes ``[0, rotary_dim)``, pairing ``j`` with
        ``j + rotary_dim // 2``, and passes ``[rotary_dim, head_dim)`` through.
        The fused store pairs lane ``i`` with ``i + head_dim // 2`` across the
        whole head. Interleaving the two blocks at their midpoints -- rotated
        first half, pass-through first half, rotated second half, pass-through
        second half -- makes every kernel pair either one reference pair or two
        pass-through lanes, which the table's identity angles leave alone.

        Attention is unaffected: ``q . k`` is invariant under a permutation
        shared by q and k, and V is never permuted, so the attention output
        stays in checkpoint lane order for ``o_proj``.

        Args:
            x: Flat ``[*, num_heads * head_dim]`` activations.
            num_heads: Number of heads packed into ``x``.
            head_dim: Lanes per head.

        Returns:
            The permuted activations, in the same flat layout.
        """
        half_rotated = self.rotary_dim // 2
        half_passed = (head_dim - self.rotary_dim) // 2
        heads = x.reshape((-1, num_heads, head_dim))
        rotated_lo = heads[:, :, :half_rotated]
        rotated_hi = heads[:, :, half_rotated : 2 * half_rotated]
        passed_lo = heads[:, :, self.rotary_dim : self.rotary_dim + half_passed]
        passed_hi = heads[:, :, self.rotary_dim + half_passed :]
        permuted = ops.concat(
            (rotated_lo, passed_lo, rotated_hi, passed_hi), axis=-1
        )
        return permuted.reshape((-1, num_heads * head_dim))

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Runs the forward pass through Laguna attention.

        Uses the fused ``rope_split_store_ragged`` store path rather than the
        unfused ``fused_qkv_ragged_matmul`` + in-cache ``rms_norm_key_cache`` +
        ``fused_qk_ragged_rope``. The unfused store kernels are monomorphic on
        the cache dtype, so they cannot write bf16-computed K/V into an FP8
        ``kv_blocks`` buffer; the fused store converts ``bf16 -> cache_dtype`` at
        store time, which is what lets the experimental
        ``--kv-cache-format float8_e4m3fn`` path run (FP8 KV is not yet
        accuracy-validated; the default bf16 KV cache is the validated path).
        Q/K RMSNorm is therefore applied per-head *before* the store — equivalent
        to the previous in-cache K-norm, since both normalise each [head_dim]
        vector.

        ``layer_idx`` is the layer's index *within its own cache leaf*, which
        is what the store and attention kernels address; it is not the layer's
        position in the model.
        """
        total_seq_len = x.shape[0]
        head_dim = self.kv_params.head_dim

        # QKV projections. Attention weights are bf16 (in the NVFP4 quant
        # ``ignore`` list), so these are dense matmuls; a quantized Linear would
        # dequant internally and still return bf16 activations. Assemble a flat
        # [total_seq_len, q_dim + k_dim + v_dim] buffer for the fused kernel.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Per-head Q/K RMSNorm before the store.
        q = self._per_head_rms_norm(q, self.q_norm, self.n_heads, head_dim)
        k = self._per_head_rms_norm(
            k, self.k_norm, self.num_key_value_heads, head_dim
        )

        # After the norms, so the checkpoint's [head_dim] gammas stay in
        # checkpoint lane order.
        if self.rotary_dim < head_dim:
            q = self._permute_rotary_lanes(q, self.n_heads, head_dim)
            k = self._permute_rotary_lanes(
                k, self.num_key_value_heads, head_dim
            )

        qkv = ops.concat((q, k, v), axis=-1)

        # Fused RoPE + KV store. For an FP8 cache, emit the roped Q in the cache
        # dtype so ``flash_attention_ragged``'s ``input.dtype == kv_params.dtype``
        # guard passes (Q and the cached K are both FP8 for the QK^T dot). For a
        # bf16/fp16/fp32 cache this stays None (Q keeps the activation dtype).
        q_out_dtype = (
            self.kv_params.dtype if self.kv_params.is_fp8_kv_dtype else None
        )
        freqs_cis = ops.cast(freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
            q_out_dtype=q_out_dtype,
        )
        xq = xq.reshape((-1, self.n_heads, head_dim))

        # Flash attention (causal). ``output_dtype`` is pinned to the
        # activation dtype so an FP8 query still yields a bf16 attention output
        # for the gate multiply and o_proj.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=(
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if self.is_sliding
                else MHAMaskVariant.CAUSAL_MASK
            ),
            local_window_size=self.local_window_size if self.is_sliding else -1,
            scale=self.scale,
            output_dtype=self.dtype,
        )

        # Softplus output gate, computed on the pre-attention hidden states
        # ``x``. HF accumulates it in float32 and casts back:
        #     gate = F.softplus(g_proj(hidden_states).float()).to(dtype)
        gate = ops.cast(
            _softplus(ops.cast(self.g_proj(x), DType.float32)), attn_out.dtype
        )
        if self.gating_mode == "per-head":
            # gate is [T, n_heads]; broadcast one scalar over each head's lanes.
            gated = attn_out.reshape(
                (total_seq_len, self.n_heads, head_dim)
            ) * ops.unsqueeze(gate, -1)
            attn_out = ops.reshape(gated, shape=[total_seq_len, -1])
        else:
            attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1]) * gate
        return self.o_proj(attn_out)
