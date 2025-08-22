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
"""Pipeline-parallel (PP) Llama 3 implementation.

This module implements pipeline parallelism by decomposing transformer layers
across multiple GPU stages, with each stage owning a contiguous slice of layers.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from math import ceil
from typing import TYPE_CHECKING, Any, Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.graph.quantization import QuantizationEncoding
from max.nn import (
    MLP,
    AttentionWithRope,
    Embedding,
    GGUFQAttentionWithRope,
    GPTQAttentionWithRope,
    GPTQLinear,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    RMSNorm,
    Transformer,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheStrategy,
)
from max.nn.transformer import ReturnLogits

# Import shared classes from llama3.py to avoid duplication
from .llama3 import ConstantLayerNorm, StackedMLP

if TYPE_CHECKING:
    from .model_config import Llama3Config

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline Parallel KV Cache Collections - MOVED TO model.py
# =============================================================================

# KV cache classes have been moved to model.py for better architecture


class PipelineParallelLlama3(Transformer):
    """Pipeline-parallel Llama3 model with stage-specific KV cache management.

    This model shards layers across pipeline stages, with each stage managing
    its own KV cache for the layers it owns.
    """

    def __init__(self, config: Llama3Config):
        """Initialize the pipeline parallel model.

        Args:
            config: Model configuration with pipeline_parallel_degree > 1
        """
        pp_degree = config.pipeline_parallel_degree
        if pp_degree <= 1:
            raise ValueError(
                f"Pipeline parallel degree must be > 1, got {pp_degree}"
            )

        if len(config.devices) != pp_degree:
            raise ValueError(
                f"Expected {pp_degree} devices, got {len(config.devices)}"
            )

        logger.info(
            f"[PP Debug] Initializing Pipeline Parallel Llama3 with {pp_degree} stages"
        )
        logger.debug(f"[PP Debug] Devices: {[d.id for d in config.devices]}")

        # Store pipeline parallel configuration
        self.pp_degree = pp_degree
        self.devices = [DeviceRef(d.device_type, d.id) for d in config.devices]
        self.stage_assignments = self._compute_stage_assignments(
            config.num_hidden_layers, pp_degree
        )

        logger.debug("[PP Debug] Layer distribution:")
        for stage_idx, (start, end) in enumerate(self.stage_assignments):
            logger.debug(
                f"[PP Debug]   Stage {stage_idx} (layers {start}-{end - 1}): {self.devices[stage_idx]}"
            )

        # Create all model components first
        layers, norm, lm_head, embed_tokens = self._create_model_components(
            config
        )

        # Select KV collection class (like DistributedLlama3)
        kv_collection_cls: type[FetchPagedKVCacheCollection]
        if config.kv_params.cache_strategy == KVCacheStrategy.PAGED:
            kv_collection_cls = FetchPagedKVCacheCollection
        else:
            raise ValueError(
                f"Unsupported cache strategy: {config.kv_params.cache_strategy}"
            )

        # Call parent constructor with simple KV collection (like other models)
        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,  # Pass layers list directly
            norm=norm,
            output=lm_head,
            embedding=embed_tokens,
            kv_params=config.kv_params,
            kv_collection_constructor=kv_collection_cls(
                config.kv_params, num_layers=config.num_hidden_layers
            ),
            return_logits=config.return_logits,
            embedding_multiplier=config.embedding_multiplier,
            rope=self.rope,
            logits_postprocessor=config.logits_postprocessor,
        )

        logger.debug(
            "[PP Debug] Pipeline parallel model created with simplified KV cache"
        )

    @staticmethod
    def _compute_stage_assignments(
        num_layers: int, pp_degree: int
    ) -> list[tuple[int, int]]:
        """Compute which layers belong to which pipeline stage."""
        layers_per_stage = ceil(num_layers / pp_degree)
        assignments = []

        for stage in range(pp_degree):
            start = stage * layers_per_stage
            end = min((stage + 1) * layers_per_stage, num_layers)
            if start < end:
                assignments.append((start, end))

        return assignments

    def _create_model_components(self, config: Llama3Config):
        """Create embedding, layers, norm, and output components using standard approach."""
        # Use the same approach as llama3.py to avoid weight naming issues
        self.rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
            device=self.devices[0],  # Use first device for rope
        )

        # Select norm layer class
        create_norm: Callable[..., Module]
        if config.norm_method == "rms_norm":
            if config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.hidden_size,
                config.norm_dtype or config.dtype,
                config.rms_norm_eps,
                multiply_before_cast=False,
            )
        else:
            create_norm = functools.partial(
                ConstantLayerNorm,
                config.hidden_size,
                self.devices[-1],  # Put norm on last stage device
                config.norm_dtype or config.dtype,
            )

        # Select linear layer class
        linear_cls: Callable[..., Linear]
        if config.quantization_config:
            linear_cls = functools.partial(
                GPTQLinear, quantization_config=config.quantization_config
            )
        else:
            linear_cls = functools.partial(
                Linear, float8_config=config.float8_config
            )

        if config.stacked_mlp and config.float8_config:
            raise ValueError("StackedMLP and float8 are not compatible")

        mlp_cls = (
            StackedMLP
            if config.stacked_mlp
            else functools.partial(MLP, float8_config=config.float8_config)
        )

        attention_cls: Callable[..., AttentionWithRope]
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            assert config.quantization_config is not None
            assert not config.attention_bias
            attention_cls = functools.partial(
                GPTQAttentionWithRope,
                quantization_config=config.quantization_config,
                scale=config.attention_multiplier,
            )
        elif config.model_quantization_encoding is not None:
            assert not config.attention_bias
            attention_cls = functools.partial(
                GGUFQAttentionWithRope,
                quantization_encoding=config.model_quantization_encoding,
                scale=config.attention_multiplier,
            )
        else:
            attention_cls = functools.partial(
                AttentionWithRope,
                stacked_qkv=config.stacked_qkv,
                scale=config.attention_multiplier,
                clip_qkv=config.clip_qkv,
                has_bias=config.attention_bias,
                float8_config=config.float8_config,
            )

        # Create transformer layers with device assignment - same as llama3.py but with device placement
        layers = []
        for i in range(config.num_hidden_layers):
            stage_id = self._get_stage_for_layer(i, config.num_hidden_layers)
            layer_device = self.devices[stage_id]

            # Create layer with device-specific placement
            layer = TransformerBlock(
                attention=attention_cls(
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    dtype=config.dtype,
                    rope=self.rope,
                    linear_cls=linear_cls,
                    devices=[layer_device],
                ),
                mlp=mlp_cls(
                    config.dtype,
                    config.model_quantization_encoding,
                    config.hidden_size,
                    config.intermediate_size,
                    [layer_device],
                    linear_cls,
                ),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=config.residual_multiplier,
            )
            layers.append(layer)

        # Create embedding and output layers
        embedding_output_dtype = config.dtype
        embedding_output_quantization = config.model_quantization_encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            embedding_output_dtype = DType.bfloat16
            embedding_output_quantization = None
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            embedding_output_dtype,
            self.devices[0],  # Embedding on first device
            quantization_encoding=embedding_output_quantization,
        )

        self.lm_head = Linear(
            config.hidden_size,
            config.vocab_size,
            embedding_output_dtype,
            self.devices[-1],  # Output on last device
            quantization_encoding=embedding_output_quantization,
        )

        # TODO: Implement proper tied weights for pipeline parallelism
        # For now, disable tied weights since embedding (device 0) and output (device N-1)
        # are on different devices, which would cause weight name collisions in the graph
        if config.tie_word_embeddings:
            logger.warning(
                "[PP] Tied word embeddings not yet supported in pipeline parallelism. "
                "Using separate weights for embedding and output layers."
            )

        self.norm = create_norm()

        return layers, self.norm, self.lm_head, self.embed_tokens

    def _get_stage_for_layer(self, layer_idx: int, num_layers: int) -> int:
        """Get which stage a layer belongs to using O(1) arithmetic."""
        # Calculate layers per stage (same as in _compute_stage_assignments)
        layers_per_stage = ceil(num_layers / self.pp_degree)
        return layer_idx // layers_per_stage

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_collections: Sequence[
            Any
        ],  # KV cache collection objects, not tensors
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Simplified pipeline parallel forward pass with pre-built KV cache collections.

        Args:
            tokens: Input tokens
            kv_cache_collections: Pre-built KV cache collections for each stage
            return_n_logits: Number of logits to return
            input_row_offsets: Row offsets for batched inputs
        """

        if len(kv_cache_collections) != self.pp_degree:
            raise ValueError(
                f"Expected {self.pp_degree} KV cache collections, got {len(kv_cache_collections)}"
            )

        # Embedding on first device
        h = self.embed_tokens(tokens)
        assert h is not None, "Embedding should always return a valid tensor"

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        # Copy input_row_offsets to all stage devices
        input_row_offsets_per_stage = []
        for stage_idx in range(self.pp_degree):
            stage_device = self.devices[stage_idx]
            input_row_offsets_per_stage.append(
                ops.transfer_to(input_row_offsets, stage_device)
            )

        # Precompute rope freqs_cis.
        freqs_cis = self.rope.freqs_cis

        # Execute pipeline stages sequentially
        current_device = self.devices[0]
        for stage_idx, (start_layer, end_layer) in enumerate(
            self.stage_assignments
        ):
            target_device = self.devices[stage_idx]

            # Transfer hidden states to stage device if needed
            if target_device != current_device:
                assert h is not None, (
                    "Hidden states should not be None during pipeline processing"
                )
                h = ops.transfer_to(h, target_device)
                current_device = target_device

            # Process layers in this stage using pre-built KV collection
            stage_kv_collection = kv_cache_collections[stage_idx]
            stage_input_row_offsets = input_row_offsets_per_stage[stage_idx]

            for layer_idx in range(start_layer, end_layer):
                # Use LOCAL layer index for KV cache access
                local_layer_idx = layer_idx - start_layer

                h = self.layers[layer_idx](
                    ops.constant(
                        local_layer_idx, DType.uint32, device=DeviceRef.CPU()
                    ),
                    h,
                    stage_kv_collection,
                    freqs_cis,
                    stage_input_row_offsets,
                )

        # Final processing on last stage device
        assert h is not None, (
            "Hidden states should not be None after processing"
        )
        last_device = self.devices[-1]
        if current_device != last_device:
            h = ops.transfer_to(h, last_device)
            current_device = last_device

        # Ensure input_row_offsets is on the same device as h for final processing
        if input_row_offsets.device != h.device:
            input_row_offsets = ops.transfer_to(input_row_offsets, h.device)

        # Apply final norm and get last token logits
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h, last_token_indices, axis=0)
        last_logits = ops.cast(
            self.lm_head(self.norm(last_token_h)), DType.float32
        )

        # Handle different return logit modes (same as parent Transformer)
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            last_tokens = ops.gather(h, last_indices, axis=0)
            logits = ops.cast(
                self.lm_head(self.norm(last_tokens)), DType.float32
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h)), DType.float32)
            offsets = input_row_offsets

        # Apply logits postprocessor
        if logits is not None:
            last_logits, logits = self._apply_logits_postprocessor(
                (last_logits, logits)
            )
        else:
            last_logits = self._apply_logits_postprocessor((last_logits,))[0]

        # Transfer outputs back to first device for sampling compatibility
        first_device = self.devices[0]
        if last_logits.device != first_device:
            last_logits = ops.transfer_to(last_logits, first_device)
        if logits is not None and logits.device != first_device:
            logits = ops.transfer_to(logits, first_device)
        if offsets is not None and offsets.device != first_device:
            offsets = ops.transfer_to(offsets, first_device)

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)

    def _apply_logits_postprocessor(
        self, output: tuple[TensorValue, ...]
    ) -> tuple[TensorValue, ...]:
        if self.logits_postprocessor is None:
            return output
        return tuple(self.logits_postprocessor(elem) for elem in output)
