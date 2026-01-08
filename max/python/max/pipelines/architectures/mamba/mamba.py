# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Qwerky AI Inc. All rights reserved.
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
"""Build a Mamba model for state-space sequence modeling"""

from __future__ import annotations

from functools import partial
from collections.abc import Callable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorType, TensorValue, ops
from max.nn import (
    ConstantLayerNorm,
    Embedding,
    LayerList,
    Linear,
    Module,
    LayerNorm,
    FusedRMSNorm,
    Identity,
    GatedMLP,
    Conv1D
)
from max.nn.mamba import Block, MambaSSM
from max.pipelines.lib.lora import LoRAManager

from .model_config import MambaConfig


class Mamba(Module):
    def __init__(self, config: MambaConfig) -> None:
        assert len(config.devices) == 1
        super().__init__()
        self.config = config

        # Select norm layer class.
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = partial(
                FusedRMSNorm,
                config.hidden_size,
                config.norm_dtype or config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            )
        else:
            create_norm = partial(
                ConstantLayerNorm,
                config.hidden_size,
                config.devices[0],
                config.norm_dtype or config.dtype,
            )

        # Select linear layer class.
        linear_cls: Callable[..., Linear]

        linear_cls = partial(
            Linear, float8_config=config.float8_config
        )

        # Create Mamba blocks
        layers = []
        for i in range(config.num_hidden_layers):
            ssm = MambaSSM(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dtype=config.dtype,
                device=config.devices[0],
                linear_cls=linear_cls,
                d_state=config.d_state,
                dt_rank=config.dt_rank if config.dt_rank is not None else "auto",
                conv_bias=config.use_conv_bias,
                bias=config.use_bias,
                delta_softplus=True,  # Always True for Mamba
                conv_kernel=config.conv_kernel,
                x_proj_dim=config.x_proj_dim,
                dt_min=config.time_step_min,
                dt_max=config.time_step_max,
                dt_init=config.time_step_init_scheme,
                dt_scale=config.time_step_scale,
                dt_init_floor=config.time_step_floor,
                use_fast_path=False,  # Disabled: fast path has bugs with multi-token sequences
                layer_idx=i,  # Pass layer index for inference caching
            )
            layers.append(
                Block(
                    dim=config.hidden_size,
                    mixer=ssm,
                    mlp=None,  # TODO: Add MLP support if needed
                    norm=create_norm(),
                    norm2=None,
                    fused_add_norm=True,  # Enable fused path
                    residual_in_fp32=config.residual_in_fp32,
                )
            )

        # Create Embedding and output layers.
        embedding_output_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype

        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices[0],
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            config.devices[0],
        )

        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        # Initialize the model with Mamba architecture
        self.embedding = embedding_layer
        self.output = output
        self.layers = LayerList(layers)
        self.norm = create_norm()
        self.hidden_size = config.hidden_size
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states

    def __call__(
        self,
        tokens: TensorValue,
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        """Forward pass through the Mamba model.

        Args:
            tokens: Input token IDs.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged tensor processing.

        Returns:
            Tuple of output tensors (logits, and optionally offsets and hidden states).
        """
        # Embed tokens
        h = self.embedding(tokens)
        
        # Process through Mamba blocks
        # Block returns (hidden_states, residual), so we need to handle that
        # Our Block returns (mixer_output, residual) where residual accumulates sums
        # but hidden_states is just the mixer output. The residual add happens
        # at the start of the next layer.
        residual = None
        for layer in self.layers:
            h, residual = layer(
                hidden_states=h,
                residual=residual,
                input_row_offsets=input_row_offsets,
            )
        
        # After all layers, h is the last mixer output and residual contains the
        # accumulated sum. We need to add them before the final norm.
        # This matches HuggingFace's MambaBlock which does: hidden_states = residual + hidden_states
        if residual is not None:
            h = h + residual
        
        # Apply final normalization
        h = self.norm(h)
        
        # Get last token logits
        last_token_indices = input_row_offsets[1:] - 1
        last_h = ops.gather(h, last_token_indices, axis=0)
        last_logits = ops.cast(self.output(last_h), DType.float32)
        
        # TODO: Implement variable logits and hidden states return logic
        # similar to Transformer class if needed
        
        return (last_logits,)

    def input_types(
        self,
        lora_manager: LoRAManager | None,
        needs_hidden_state_input: bool = False,
    ) -> tuple[TensorType, ...]:
        # TODO: Move input symbol computation from the manager classes.
        device_ref = self.config.devices[0]

        # Construct general input types
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        # Construct Graph Inputs
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )

        if lora_manager is not None:
            (
                lora_ids,
                lora_ranks,
                lora_grouped_offsets,
                num_active_loras,
                lora_end_idx,
                batch_seq_len,
                lora_ids_kv,
                lora_grouped_offsets_kv,
            ) = lora_manager.get_symbolic_inputs(device_ref)
            # Add seqlen_offset for SSM state caching
            seqlen_offset_type = TensorType(
                DType.int32, shape=["seqlen_offset"], device=DeviceRef.CPU()
            )

            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                lora_ids,
                lora_ranks,
                lora_grouped_offsets,
                num_active_loras,
                lora_end_idx,
                batch_seq_len,
                lora_ids_kv,
                lora_grouped_offsets_kv,
            )

        if needs_hidden_state_input:
            hidden_states_type = TensorType(
                self.config.dtype,
                shape=["total_seq_len", self.config.hidden_size],
                device=device_ref,
            )
            # Add seqlen_offset for SSM state caching
            seqlen_offset_type = TensorType(
                DType.int32, shape=["seqlen_offset"], device=DeviceRef.CPU()
            )

            return (
                tokens_type,
                input_row_offsets_type,
                return_n_logits_type,
                hidden_states_type,
            )

        # Add seqlen_offset for SSM state caching
        seqlen_offset_type = TensorType(
            DType.int32, shape=["seqlen_offset"], device=DeviceRef.CPU()
        )

        return (
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        )

