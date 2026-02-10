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
"""Build a Nemotron model with partial RoPE, LayerNorm, and squared ReLU MLP."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.nn.legacy.attention import AttentionWithRope
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.kv_cache import KVCacheParams
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import LayerNorm
from max.nn.legacy.rotary_embedding import RotaryEmbedding
from max.nn.legacy.transformer import Transformer, TransformerBlock
from max.pipelines.lib.lora import LoRAManager

from .model_config import NemotronConfig


class PartialRotaryEmbedding(RotaryEmbedding):
    """Rotary embedding that applies RoPE to only a fraction of head dimensions.

    Nemotron uses ``partial_rotary_factor=0.5``, meaning only the first half of
    each head's dimensions receive rotary positional encodings.  The remaining
    dimensions pass through unchanged (identity rotation: cos=1, sin=0).
    """

    partial_rotary_factor: float

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        partial_rotary_factor: float = 0.5,
        head_dim: int | None = None,
        interleaved: bool = True,
    ) -> None:
        super().__init__(
            dim=dim,
            n_heads=n_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            interleaved=interleaved,
        )
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)

    def _compute_inv_freqs(self) -> TensorValue:
        """Computes inverse frequencies for the rotary dimensions only.

        The denominator uses ``rotary_dim`` (not ``head_dim``) so frequency
        spacing matches what HuggingFace's Nemotron implementation produces.
        """
        n = self.rotary_dim
        iota = ops.range(
            0, n, step=2, dtype=DType.float64, device=DeviceRef.CPU()
        )
        inv_freq = ops.cast(1.0 / (self.theta ** (iota / n)), DType.float32)
        return inv_freq

    def freqs_cis_base(self) -> TensorValue:
        """Computes frequency tensor padded with identity for non-rotary dims.

        Returns:
            Tensor of shape ``(max_seq_len * 2, head_dim // 2, 2)`` where
            the first ``rotary_dim // 2`` pairs carry real rotation angles
            and the rest are ``(cos=1, sin=0)`` (identity).
        """
        if self._freqs_cis is None:
            inv_freqs = self._compute_inv_freqs()  # [rotary_dim // 2]

            t = ops.range(
                0,
                self.max_seq_len * 2,
                device=DeviceRef.CPU(),
                dtype=DType.float32,
            )
            freqs = ops.outer(t, inv_freqs)  # [max_seq_len*2, rotary_dim//2]

            cos_freqs = ops.cos(freqs)
            sin_freqs = ops.sin(freqs)

            # Pad non-rotary dimensions with identity rotation (cos=1, sin=0).
            non_rotary_pairs = (self.head_dim - self.rotary_dim) // 2
            if non_rotary_pairs > 0:
                seq_len_dim = cos_freqs.shape[0]
                ones_pad = ops.broadcast_to(
                    ops.constant(1.0, DType.float32, DeviceRef.CPU()),
                    shape=(seq_len_dim, non_rotary_pairs),
                )
                zeros_pad = ops.broadcast_to(
                    ops.constant(0.0, DType.float32, DeviceRef.CPU()),
                    shape=(seq_len_dim, non_rotary_pairs),
                )
                cos_freqs = ops.concat(
                    [cos_freqs, ones_pad], axis=-1
                )  # [seq*2, head_dim//2]
                sin_freqs = ops.concat(
                    [sin_freqs, zeros_pad], axis=-1
                )  # [seq*2, head_dim//2]

            self._freqs_cis = ops.stack(
                [cos_freqs, sin_freqs], axis=-1
            )  # [max_seq_len*2, head_dim//2, 2]
        return TensorValue(self._freqs_cis)


class NemotronMLP(Module):
    """Simple (non-gated) MLP with squared ReLU activation.

    Unlike the LLaMA-style gated MLP (``gate * up``), Nemotron uses a standard
    two-layer MLP::

        output = down_proj(squared_relu(up_proj(x)))

    where ``squared_relu(x) = relu(x) ** 2``.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.up_proj = linear_cls(
            in_dim=hidden_dim,
            out_dim=feed_forward_length,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )
        self.down_proj = linear_cls(
            in_dim=feed_forward_length,
            out_dim=hidden_dim,
            dtype=dtype,
            device=devices[0],
            quantization_encoding=quantization_encoding,
        )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Forward pass: up_proj -> squared_relu -> down_proj."""
        h = self.up_proj(x)
        h = ops.relu(h)
        h = h * h  # squared ReLU
        return self.down_proj(h)


class Nemotron(Transformer):
    """Nemotron causal language model (NemotronForCausalLM).

    Key architectural differences from LLaMA:
    - LayerNorm (with learned weight + bias) instead of RMSNorm.
    - Non-gated MLP with squared ReLU activation.
    - Partial rotary positional embeddings (``partial_rotary_factor=0.5``).
    - Explicit ``kv_channels`` (head_dim) that may differ from
      ``hidden_size // num_attention_heads``.
    """

    def __init__(self, config: NemotronConfig) -> None:
        assert len(config.devices) == 1
        self.config = config

        # --- Partial Rotary Embedding ---
        rope = PartialRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            partial_rotary_factor=config.partial_rotary_factor,
            head_dim=config.head_dim,
            interleaved=config.interleaved_rope_weights,
        )

        # --- LayerNorm ---
        create_norm: Callable[..., Module] = functools.partial(
            LayerNorm,
            config.hidden_size,
            config.devices,
            config.norm_dtype or config.dtype,
            eps=config.norm_eps,
        )

        # --- Linear layer class ---
        linear_cls: Callable[..., Linear] = functools.partial(
            Linear, float8_config=config.float8_config
        )

        # --- Attention layers ---
        attention_cls: Callable[..., AttentionWithRope] = functools.partial(
            AttentionWithRope,
            stacked_qkv=config.stacked_qkv,
            scale=config.attention_multiplier,
            has_bias=config.attention_bias,
            float8_config=config.float8_config,
        )

        # --- Transformer blocks ---
        layers = [
            TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=rope,
                    linear_cls=linear_cls,
                    devices=config.devices,
                ),
                mlp=NemotronMLP(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    config.devices,
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            for _ in range(config.num_hidden_layers)
        ]

        # --- Embedding & output projection ---
        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices[0],
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            config.dtype,
            config.devices[0],
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
            embedding_multiplier=config.embedding_multiplier,
        )

    def input_types(
        self,
        kv_params: KVCacheParams,
        lora_manager: LoRAManager | None = None,
        needs_hidden_state_input: bool = False,
    ) -> tuple[TensorType, ...]:
        """Constructs symbolic input types for graph compilation."""
        device_ref = self.config.devices[0]

        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = kv_params.get_symbolic_inputs()

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *kv_inputs[0],
        )
