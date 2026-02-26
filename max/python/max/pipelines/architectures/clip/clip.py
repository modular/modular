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

from functools import partial

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import ShapeLike, TensorType
from max.nn.module_v3 import Embedding, Linear, Module
from max.nn.module_v3.norm import LayerNorm
from max.nn.module_v3.sequential import ModuleList

from .model_config import ClipConfigBase


class CLIPTextEmbeddings(Module[..., Tensor]):
    def __init__(
        self,
        config: ClipConfigBase,
    ):
        """Initialize CLIP text embeddings.

        Args:
            config: CLIP configuration for embedding dimensions and device/dtype.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.position_embedding = Embedding(
            config.max_position_embeddings,
            dim=self.embed_dim,
        )
        self.token_embedding = Embedding(
            config.vocab_size,
            dim=self.embed_dim,
        )

    def forward(
        self,
        input_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        """Apply embeddings to input tokens.

        Args:
            input_ids: Input token IDs.
            position_ids: Position IDs.
            inputs_embeds: Pre-computed input embeddings.

        Returns:
            Combined embeddings.
        """

        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )

        if position_ids is None:
            if input_ids is not None:
                device = input_ids.device
            else:
                assert inputs_embeds is not None
                device = inputs_embeds.device
            position_ids = F.arange(
                0, seq_length, step=1, dtype=DType.int32, device=device
            )
            position_ids = F.unsqueeze(position_ids, 0)

        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(Module[..., Tensor]):
    def __init__(
        self,
        config: ClipConfigBase,
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

        self.k_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )
        self.v_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )
        self.q_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )
        self.out_proj = Linear(
            self.embed_dim,
            self.embed_dim,
            bias=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        causal_attention_mask: Tensor | None = None,
    ) -> Tensor:
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

        query = F.reshape(
            query, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        query = F.transpose(query, 1, 2)

        key = F.reshape(
            key, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        key = F.transpose(key, 1, 2)

        value = F.reshape(
            value, (batch_size, seq_length, self.num_heads, self.head_dim)
        )
        value = F.transpose(value, 1, 2)

        if attention_mask is not None and causal_attention_mask is not None:
            attention_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attention_mask = causal_attention_mask

        attn_weights = F.matmul(query, F.transpose(key, -1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(F.cast(attn_weights, DType.float32), axis=-1)
        attn_weights = F.cast(attn_weights, hidden_states.dtype)

        attn_output = F.matmul(attn_weights, value)
        attn_output = F.transpose(attn_output, 1, 2)
        attn_output = F.reshape(
            attn_output, (batch_size, seq_length, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(Module[[Tensor], Tensor]):
    def __init__(
        self,
        config: ClipConfigBase,
    ):
        """Initialize CLIP MLP.

        Args:
            config: CLIP configuration for MLP dimensions and device/dtype.
        """
        super().__init__()
        self.config = config
        self.fc1 = Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
        )
        self.fc2 = Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
        )
        self.act_fn = partial(F.gelu, approximate="quick")

    def forward(self, hidden_states: Tensor) -> Tensor:
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


class CLIPEncoderLayer(Module[..., Tensor]):
    def __init__(
        self,
        config: ClipConfigBase,
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
            keep_dtype=True,
        )
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=config.layer_norm_eps,
            keep_dtype=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        causal_attention_mask: Tensor | None,
    ) -> Tensor:
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
            hidden_states,
            attention_mask,
            causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(Module[..., Tensor]):
    def __init__(
        self,
        config: ClipConfigBase,
    ):
        """Initialize CLIP encoder.

        Args:
            config: CLIP configuration for encoder depth and dimensions.
        """
        super().__init__()
        self.layers = ModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        causal_attention_mask: Tensor | None = None,
    ) -> Tensor:
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
                attention_mask,
                causal_attention_mask,
            )
        return hidden_states


class CLIPTextTransformer(Module[..., tuple[Tensor, Tensor]]):
    def __init__(
        self,
        config: ClipConfigBase,
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
            keep_dtype=True,
        )
        self.eos_token_id = config.eos_token_id

    def _create_causal_mask(
        self, input_shape: ShapeLike, device: Device
    ) -> Tensor:
        """Create causal mask for the transformer.

        Args:
            input_shape: Shape of the input tensor.

        Returns:
            Causal mask tensor.
        """
        _, seq_length = input_shape

        rows = F.arange(0, seq_length, step=1, dtype=DType.int32, device=device)
        rows = F.unsqueeze(rows, 1)
        cols = F.arange(0, seq_length, step=1, dtype=DType.int32, device=device)
        cols = F.unsqueeze(cols, 0)
        mask = F.greater(cols, rows)
        mask_float = F.cast(mask, self.config.dtype)

        min_val = DType.finfo(self.config.dtype).min  # type: ignore[attr-defined]
        min_val_tensor = F.constant(
            min_val, dtype=self.config.dtype, device=device
        )

        causal_mask = mask_float * min_val_tensor
        causal_mask = F.unsqueeze(causal_mask, 0)
        causal_mask = F.unsqueeze(causal_mask, 1)
        return causal_mask

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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
        causal_attention_mask = self._create_causal_mask(
            input_shape, input_ids.device
        )

        if attention_mask is not None:
            mask_multiplier = F.constant(
                DType.finfo(hidden_states.dtype).min,  # type: ignore[attr-defined]
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            inverted_mask = (
                F.constant(
                    1.0,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                - F.cast(attention_mask, hidden_states.dtype)
            ) * mask_multiplier
            attention_mask = F.unsqueeze(inverted_mask, 1)
            attention_mask = F.unsqueeze(attention_mask, 1)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = self.final_layer_norm(encoder_outputs)

        if self.eos_token_id == 2:
            eos_token_indices = F.cast(
                F.argmax(input_ids, axis=-1), DType.int32
            )
        else:
            eos_token_indices = F.cast(
                F.argmax(
                    F.cast(F.equal(input_ids, self.eos_token_id), DType.int32),
                    axis=-1,
                ),
                DType.int32,
            )

        pooled_output = F.gather_nd(
            last_hidden_state, eos_token_indices, batch_dims=1
        )

        return last_hidden_state, pooled_output


class CLIPTextModel(Module[..., tuple[Tensor, Tensor]]):
    def __init__(
        self,
        config: ClipConfigBase,
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

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
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
