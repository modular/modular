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
"""Build a Mamba model that runs on multiple devices."""

from __future__ import annotations

import functools
import logging
from collections import defaultdict

from max.dtype import DType
from max.graph import BufferType, BufferValue, DeviceRef, TensorType, TensorValue, Value, ops
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    ColumnParallelLinear,
    FusedRMSNorm,
    ConstantLayerNorm,
    LayerList,
    Linear,
    Module,
    Signals,
    VocabParallelEmbedding,
)
from max.nn.mamba import Block, MambaSSM
from functools import partial
from typing import Callable

logger = logging.getLogger("max.pipelines")
from .model_config import MambaConfig


class DistributedMamba(Module):
    def __init__(self, config: MambaConfig) -> None:
        assert len(config.devices) > 1
        self.config = config

        # For now, implement a basic distributed Mamba that creates the same
        # architecture as single-device Mamba but can handle signal buffers.
        # True tensor parallelism for Mamba would require more complex implementation
        # since Mamba's SSM state creates sequential dependencies.

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
                config.devices[0],  # Use first device for now
                config.norm_dtype or config.dtype,
            )

        # Select linear layer class - use regular Linear for now since tensor parallelism
        # for Mamba is complex due to sequential SSM dependencies
        linear_cls = partial(
            Linear,
            float8_config=config.float8_config
        )

        # Create Mamba blocks
        layers = []
        for i in range(config.num_hidden_layers):
            ssm = MambaSSM(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                dtype=config.dtype,
                device=config.devices[0],  # Use first device for now
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
                use_fast_path=True,  # Use fast path when available
                layer_idx=i,  # Pass layer index for inference caching
            )
            layers.append(
                Block(
                    dim=config.hidden_size,
                    mixer=ssm,
                    mlp=None,  # TODO: Add MLP support if needed
                    norm=create_norm(),
                    norm2=None,
                    fused_add_norm=config.fused_add_norm,
                    residual_in_fp32=config.residual_in_fp32,
                )
            )

        self.layers = LayerList(layers)

        # Create Embedding layer
        embedding_output_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype

        self.embedding = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            config.devices,
        )

        # Create final norm layer
        self.norm = create_norm()

        # Create output layer
        self.lm_head = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices,
            transpose=True,  # For output projection
        )

    def input_types(
        self,
    ) -> tuple[TensorType | BufferType, ...]:
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

        signals = Signals(devices=self.config.devices)

        # Explicitly construct tuple with mixed types
        signal_buffer_types: list[BufferType] = signals.input_types()

        # Build the complete input types list
        all_input_types: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]
        all_input_types.extend(signal_buffer_types)

        return tuple(all_input_types)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: list[BufferValue],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Forward pass for distributed Mamba model without KV cache.

        Args:
            tokens: Input token IDs.
            signal_buffers: Signal buffers for multi-GPU communication.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged tensor processing.

        Returns:
            Tuple of output tensors (logits, and optionally offsets and hidden states).
        """
        # Embed tokens
        h = self.embedding(tokens)

        # Process through Mamba blocks
        # For now, process sequentially since Mamba has sequential dependencies
        residual = None
        for layer in self.layers:
            h, residual = layer(
                hidden_states=h,
                residual=residual,
                input_row_offsets=input_row_offsets,
                inference_params=None,  # No state caching for now
            )

        # Apply final normalization
        h = self.norm(h)

        # Get logits
        logits = self.lm_head(h)

        # Return logits for the last token(s) based on return_n_logits
        last_token_indices = input_row_offsets[1:] - 1
        if return_n_logits.shape[0] == 1 and return_n_logits[0] == 1:
            # Return only last token logits
            return (ops.gather(logits, last_token_indices, axis=0),)
        else:
            # Return all logits (this is a simplified implementation)
            return (logits,)

