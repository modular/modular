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

from functools import partial

import max.nn as nn
from max.dtype import DType
from max.graph import TensorType, TensorValue, ops
from max.nn import LayerNorm, Module

from .model_config import ClipConfig


class CLIPTextEmbeddings(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP text embeddings.

        Args:
            config: CLIP configuration for embedding dimensions and device/dtype.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings,
            self.embed_dim,
            device=config.device,
            dtype=config.dtype,
        )
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            self.embed_dim,
            device=config.device,
            dtype=config.dtype,
        )

    def __call__(
        self,
        input_ids: TensorValue | None = None,
        position_ids: TensorValue | None = None,
        inputs_embeds: TensorValue | None = None,
    ) -> TensorValue:
        """Apply embeddings to input tokens.

        Args:
            input_ids: Input token IDs.
            position_ids: Position IDs.
            inputs_embeds: Pre-computed input embeddings.

        Returns:
            Combined embeddings.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        else:
            seq_length = inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = ops.range(
                0,
                seq_length,
                step=1,
                dtype=DType.int32,
                device=self.config.device,
            )
            position_ids = ops.unsqueeze(position_ids, 0)

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP attention module.

        Args:
            config: CLIP configuration for attention dimensions and device/dtype.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            has_bias=True,
            device=config.device,
            dtype=config.dtype,
        )
        self.v_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            has_bias=True,
            device=config.device,
            dtype=config.dtype,
        )
        self.q_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            has_bias=True,
            device=config.device,
            dtype=config.dtype,
        )
        self.out_proj = nn.Linear(
            self.embed_dim,
            self.embed_dim,
            has_bias=True,
            device=config.device,
            dtype=config.dtype,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
        causal_attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        """Apply multi-head attention.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            causal_attention_mask: Causal attention mask.

        Returns:
            Attention output.
        """
        batch_size, seq_length, embed_dim = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = ops.reshape(
            query, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        query = ops.transpose(query, 1, 2)

        key = ops.reshape(
            key, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        key = ops.transpose(key, 1, 2)

        value = ops.reshape(
            value, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        value = ops.transpose(value, 1, 2)

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask

        attn_weights = (
            ops.matmul(query, ops.transpose(key, -1, -2)) * self.scale
        )

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(
            ops.cast(attn_weights, DType.float32), axis=-1
        )
        attn_weights = ops.cast(attn_weights, hidden_states.dtype)

        attn_output = ops.matmul(attn_weights, value)
        attn_output = ops.transpose(attn_output, 1, 2)
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_length, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP MLP.

        Args:
            config: CLIP configuration for MLP dimensions and device/dtype.
        """
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            has_bias=True,
            device=config.device,
            dtype=config.dtype,
        )
        self.fc2 = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            has_bias=True,
            device=config.device,
            dtype=config.dtype,
        )
        self.act_fn = partial(ops.gelu, approximate="quick")

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Apply MLP block.

        Args:
            hidden_states: Input hidden states.

        Returns:
            Output hidden states.
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP encoder layer.

        Args:
            config: CLIP configuration for encoder layer structure.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
            devices=[config.device],
            dtype=config.dtype,
            keep_dtype=True,
        )
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
            devices=[config.device],
            dtype=config.dtype,
            keep_dtype=True,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        causal_attention_mask: TensorValue,
    ) -> TensorValue:
        """Apply encoder layer.

        Args:
            hidden_states: Input hidden states.
            attention_mask: Attention mask.
            causal_attention_mask: Causal attention mask.

        Returns:
            Output hidden states.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP encoder.

        Args:
            config: CLIP configuration for encoder depth and dimensions.
        """
        super().__init__()
        self.layers = nn.LayerList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def __call__(
        self,
        inputs_embeds: TensorValue,
        attention_mask: TensorValue | None = None,
        causal_attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        """Apply encoder (stack of layers).

        Args:
            inputs_embeds: Input embeddings.
            attention_mask: Attention mask.
            causal_attention_mask: Causal attention mask.

        Returns:
            Encoded hidden states.
        """
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )
        return hidden_states


class CLIPTextTransformer(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP text transformer.

        Args:
            config: CLIP configuration for embeddings, encoder, and device/dtype.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
            devices=[config.device],
            dtype=config.dtype,
            keep_dtype=True,
        )
        self.eos_token_id = config.eos_token_id

    def _create_causal_mask(self, input_shape: tuple[int, int]) -> TensorValue:
        """Create causal mask for the transformer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Causal mask tensor.
        """
        _, seq_length = input_shape

        rows = ops.range(
            0, seq_length, step=1, dtype=DType.int32, device=self.config.device
        )
        rows = ops.unsqueeze(rows, 1)
        cols = ops.range(
            0, seq_length, step=1, dtype=DType.int32, device=self.config.device
        )
        cols = ops.unsqueeze(cols, 0)
        mask = ops.greater(cols, rows)
        mask_float = mask.cast(self.config.dtype)

        min_val = DType.finfo(self.config.dtype).min

        causal_mask = mask_float * min_val
        causal_mask = ops.unsqueeze(causal_mask, 0)
        causal_mask = ops.unsqueeze(causal_mask, 1)
        return causal_mask

    def __call__(
        self,
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
    ) -> TensorValue:
        """Apply text transformer.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.

        Returns:
            Tuple of (last_hidden_state, pooled_output).
        """
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids
        )

        input_shape = input_ids.shape
        causal_attention_mask = self._create_causal_mask(input_shape)

        if attention_mask is not None:
            inverted_mask = (
                1.0 - attention_mask.cast(hidden_states.dtype)
            ) * DType.finfo(hidden_states.dtype).min
            attention_mask = ops.unsqueeze(inverted_mask, 1)
            attention_mask = ops.unsqueeze(attention_mask, 1)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = self.final_layer_norm(encoder_outputs)

        if self.eos_token_id == 2:
            eos_token_indices = ops.argmax(input_ids, axis=-1).cast(DType.int32)
        else:
            eos_token_indices = ops.argmax(
                ops.equal(input_ids, self.eos_token_id).cast(DType.int32),
                axis=-1,
            ).cast(DType.int32)

        pooled_output = ops.gather_nd(
            last_hidden_state, eos_token_indices, batch_dims=1
        )

        return last_hidden_state, pooled_output


class CLIPTextModel(Module):
    def __init__(
        self,
        config: ClipConfig,
    ):
        """Initialize CLIP text model with MAX.

        Args:
            config: CLIP configuration for vocabulary size, dimensions, and
                device/dtype settings.
        """
        super().__init__()
        self.text_model = CLIPTextTransformer(config)
        self.device = config.device

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for the model.

        Returns:
            Tuple of TensorType specifications for model inputs.
        """
        return (
            TensorType(
                DType.int64,
                shape=["batch_size", "sequence_length"],
                device=self.device,
            ),
        )

    def __call__(
        self,
        input_ids: TensorValue | None = None,
        attention_mask: TensorValue | None = None,
        position_ids: TensorValue | None = None,
    ) -> tuple[TensorValue, TensorValue]:
        """Apply CLIP text model forward pass.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            position_ids: Position IDs.

        Returns:
            Tuple of (last_hidden_state, pooled_output).
        """
        return self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
