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

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from max.driver import Buffer, DLPackArray, is_virtual_device_mode
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferValue, DeviceRef, Graph, TensorType, TensorValue
from max.graph.buffer_utils import cast_tensors_to
from max.graph.weights import Weights, WeightsAdapter
from max.interfaces import RequestID
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
    VisionEncodingData,
)
from max.pipelines.core import TextContext
from max.pipelines.lib import (
    CompilationTimer,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    supported_encoding_dtype,
)
from max.pipelines.lib.interfaces import AlwaysSignalBuffersMixin
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.lib.vlm_utils import compute_multimodal_merge_indices
from max.profiler import traced
from transformers import AutoConfig

from ..llama3.model import Llama3Inputs, LlamaModelBase
from .model_config import Qwen3_5Config
from .qwen3_5 import Qwen3_5
from .state_cache import GatedDeltaNetStateCache

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3_5Inputs(Llama3Inputs):
    """Inputs for Qwen3.5 including linear attention states and optional vision inputs."""

    slot_idx: Buffer | None = None
    """Per-batch ``[B]`` uint32 slot indices into the linear-attention pools."""

    conv_pools: list[Buffer] | None = None
    """Per-layer mutable conv pool, ``[max_slots, conv_dim, K-1]``."""

    recurrent_pools: list[Buffer] | None = None
    """Per-layer mutable recurrent pool, ``[max_slots, nv, KD, VD]``."""

    request_ids: list[RequestID] | None = None
    """Request IDs for this batch, used to manage per-request state cache slots."""

    # Vision inputs (None for text-only or decode steps)
    image_token_indices: Buffer | None = None
    """Pre-computed scatter indices for image embeddings."""

    pixel_values: Buffer | None = None
    """Raw pixel values for vision encoding."""

    vision_position_ids: Buffer | None = None
    """Rotary position IDs for the vision encoder."""

    weights: Buffer | None = None
    """Bilinear interpolation weights for vision position embeddings."""

    indices: Buffer | None = None
    """Bilinear interpolation indices for vision position embeddings."""

    max_grid_size: Buffer | None = None
    """Maximum grid size (CPU scalar) for vision attention."""

    grid_thw: Buffer | None = None
    """Grid dimensions (temporal, height, width) per image, shape (n_images, 3)."""

    cu_seqlens: Buffer | None = None
    """Cumulative sequence lengths for vision full attention."""

    max_seqlen: Buffer | None = None
    """Maximum sequence length (CPU scalar) for vision attention."""

    lm_image_embeddings: Buffer | None = None
    """Image embeddings for the LM graph (empty [0, H] buffer for decode/text-only steps,
    real embeddings for prefill steps with images). Must be non-None for multimodal models."""

    @property
    def has_vision_inputs(self) -> bool:
        """True when pixel values are available for vision encoding."""
        return self.pixel_values is not None

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        vision_lm_inputs: tuple[Buffer, ...] = ()
        if self.lm_image_embeddings is not None:
            assert self.image_token_indices is not None
            vision_lm_inputs = (
                self.lm_image_embeddings,
                self.image_token_indices,
            )
        slot_idx_inputs: tuple[Buffer, ...] = ()
        if self.slot_idx is not None:
            slot_idx_inputs = (self.slot_idx,)
        return (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *slot_idx_inputs,
            *(self.conv_pools or ()),
            *(self.recurrent_pools or ()),
            *vision_lm_inputs,
        )


class Qwen3_5Model(AlwaysSignalBuffersMixin, LlamaModelBase):
    """Qwen3.5 pipeline model implementation.

    Supports the hybrid linear/full attention architecture with KV cache
    for full attention layers and conv/recurrent states for linear layers.
    """

    model: Model
    norm_method: Literal["rms_norm"] | Literal["layer_norm"] = "rms_norm"
    attention_bias: bool = False
    state_dict: dict[str, Any]

    # Vision model (None for text-only checkpoints)
    vision_model: Model | None = None
    _vision_state_dict: dict[str, DLPackArray] | None = None
    _nn_model: Any = None
    _session: InferenceSession | None = None

    # Model dtype and hidden size (set during graph build, used for empty buffers)
    _hidden_size: int = 0
    _model_dtype: DType = DType.bfloat16

    # Linear attention state dimensions (set during graph build)
    _num_linear_layers: int = 0
    _conv_dim: int = 0
    _conv_kernel_size: int = 0
    _num_v_heads: int = 0
    _key_head_dim: int = 0
    _value_head_dim: int = 0

    # Per-request state cache for the linear-attention pools.
    _state_cache: GatedDeltaNetStateCache | None = None

    # Pre-allocated empty vision input buffers for the LM graph (multimodal models only).
    # Used for decode/text-only steps so that buffers() always has the right input count.
    _empty_lm_image_embeddings: Buffer | None = None
    _empty_lm_image_token_indices: Buffer | None = None

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Qwen3_5Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def calculate_max_seq_len(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        return Qwen3_5Config.calculate_max_seq_len(pipeline_config, text_config)

    @classmethod
    def estimate_activation_memory(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
    ) -> int:
        """Reserve GPU memory for the GatedDeltaNet state pool.

        The slot-indexed SSM kernels mutate the conv and recurrent pools in
        place; there are no working buffers and no graph-output pool, so
        peak footprint is a single ``max_batch x per_req`` allocation.

        ``Qwen3_5Config.initialize_from_config`` pre-sets ``max_batch_size``
        before this method runs, so it is always known here.
        """
        text_config = Qwen3_5Config._get_text_config(huggingface_config)
        layer_types = Qwen3_5Config._get_layer_types(text_config)
        num_linear = sum(1 for lt in layer_types if lt == "linear_attention")

        nk = getattr(text_config, "linear_num_key_heads", 16)
        nv = getattr(text_config, "linear_num_value_heads", 48)
        kd = getattr(text_config, "linear_key_head_dim", 128)
        vd = getattr(text_config, "linear_value_head_dim", 128)
        kernel = getattr(text_config, "linear_conv_kernel_dim", 4)

        conv_dim = 2 * kd * nk + vd * nv
        # Determine state dtype bytes: states stored in model dtype (typically bfloat16).
        encoding = pipeline_config.model.quantization_encoding
        state_dtype = (
            supported_encoding_dtype(encoding)
            if encoding is not None
            else DType.bfloat16
        )
        dtype_bytes = state_dtype.size_in_bytes
        bytes_per_layer = (
            conv_dim * (kernel - 1) * dtype_bytes + nv * kd * vd * dtype_bytes
        )
        per_req = num_linear * bytes_per_layer

        max_batch = pipeline_config.runtime.max_batch_size
        assert max_batch is not None, (
            "Qwen3_5Config.initialize_from_config must set max_batch_size "
            "before estimate_activation_memory runs"
        )
        # 1x: single in-place pool — kernels mutate it via slot_idx.
        return max_batch * per_req if num_linear > 0 else 0

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        self._session = session

        self._input_row_offsets_prealloc: Buffer | None = None
        self._slot_idx_prealloc: Buffer | None = None
        max_batch_size = self.pipeline_config.runtime.max_batch_size
        assert max_batch_size is not None, (
            "max_batch_size must be set in runtime config"
        )
        if not is_virtual_device_mode():
            self._input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(
                    max_batch_size + 1,
                    dtype=np.uint32,
                )
            ).to(self.devices[0])

        with CompilationTimer("model") as timer:
            graph = self._build_graph(self.weights, self.adapter)
            timer.mark_build_complete()
            model = session.load(graph, weights_registry=self.state_dict)

        # Initialize per-request state cache for linear attention layers.
        # _num_linear_layers is populated by _build_graph, so this and the
        # slot-idx prealloc must run after it.
        if self._num_linear_layers > 0:
            self._state_cache = GatedDeltaNetStateCache(
                num_layers=self._num_linear_layers,
                conv_dim=self._conv_dim,
                conv_kernel_size=self._conv_kernel_size,
                num_v_heads=self._num_v_heads,
                key_head_dim=self._key_head_dim,
                value_head_dim=self._value_head_dim,
                max_slots=max_batch_size,
                device=self.devices[0],
                dtype=self._model_dtype,
            )
            if not is_virtual_device_mode():
                self._slot_idx_prealloc = Buffer(
                    shape=[max_batch_size],
                    dtype=DType.uint32,
                    device=self.devices[0],
                )

        if self._vision_state_dict is not None:
            # Pre-allocate empty vision input buffers for the LM graph so that
            # buffers() always returns the correct input count for CUDA graph capture.
            self._empty_lm_image_embeddings = Buffer.zeros(
                shape=[0, self._hidden_size], dtype=self._model_dtype
            ).to(self.devices[0])
            self._empty_lm_image_token_indices = Buffer.zeros(
                shape=[0], dtype=DType.int32
            ).to(self.devices[0])
            with CompilationTimer("vision model") as timer:
                vision_graph = self._build_vision_graph()
                timer.mark_build_complete()
                self.vision_model = session.load(
                    vision_graph, weights_registry=self._vision_state_dict
                )

        return model

    def _build_vision_graph(self) -> Graph:
        """Build the vision encoder graph for processing images."""
        assert isinstance(self._nn_model, Qwen3_5), (
            "_build_vision_graph called before _build_graph"
        )
        vision_encoder = self._nn_model.vision_encoder
        assert vision_encoder is not None, (
            "_build_vision_graph called but no vision encoder"
        )

        patch_dim = vision_encoder.patch_embed.patch_dim

        # Input types - one per device (currently single-device only; see arch.py)
        pixel_values_types = [
            TensorType(
                DType.float32,
                shape=["vision_seq_len", patch_dim],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        weights_types = [
            TensorType(
                DType.float32,
                shape=[4, "vision_seq_len", 1],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        indices_types = [
            TensorType(
                DType.int64,
                shape=[4, "vision_seq_len"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        rot_pos_ids_types = [
            TensorType(
                DType.int32,
                shape=["vision_seq_len", 2],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        max_grid_size_types = [
            TensorType(DType.int32, shape=[], device=DeviceRef.CPU())
            for _ in self.devices
        ]
        grid_thw_types = [
            TensorType(
                DType.int64,
                shape=["n_images", 3],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        cu_seqlens_types = [
            TensorType(
                DType.uint32,
                shape=["n_seqlens"],
                device=DeviceRef.from_device(device),
            )
            for device in self.devices
        ]
        max_seqlen_types = [
            TensorType(DType.uint32, shape=[1], device=DeviceRef.CPU())
            for _ in self.devices
        ]

        signals = Signals(
            devices=(DeviceRef(d.label, d.id) for d in self.devices)
        )

        with Graph(
            "qwen3_5_vision",
            input_types=tuple(
                [
                    *pixel_values_types,
                    *weights_types,
                    *indices_types,
                    *rot_pos_ids_types,
                    *max_grid_size_types,
                    *grid_thw_types,
                    *cu_seqlens_types,
                    *max_seqlen_types,
                    *signals.input_types(),
                ]
            ),
        ) as graph:
            all_inputs = graph.inputs
            n = len(self.devices)

            pixel_values_list = [inp.tensor for inp in all_inputs[:n]]
            weights_list = [inp.tensor for inp in all_inputs[n : 2 * n]]
            indices_list = [inp.tensor for inp in all_inputs[2 * n : 3 * n]]
            rot_pos_ids_list = [inp.tensor for inp in all_inputs[3 * n : 4 * n]]
            max_grid_size_list = [
                inp.tensor for inp in all_inputs[4 * n : 5 * n]
            ]
            grid_thw_list = [inp.tensor for inp in all_inputs[5 * n : 6 * n]]
            cu_seqlens_list = [inp.tensor for inp in all_inputs[6 * n : 7 * n]]
            max_seqlen_list = [inp.tensor for inp in all_inputs[7 * n : 8 * n]]
            signal_buffers = [inp.buffer for inp in all_inputs[8 * n :]]

            # Qwen3.5 does not use deepstack (intermediate visual features
            # injected at multiple LM depths) — that is a Qwen3VL-MoE feature.
            image_embeddings, _ = vision_encoder(
                pixel_values=pixel_values_list,
                idxs=indices_list,
                weights=weights_list,
                grid_thw=grid_thw_list,
                rot_pos_ids=rot_pos_ids_list,
                max_grid_size=max_grid_size_list,
                cu_seqlens=cu_seqlens_list,
                max_seqlen=max_seqlen_list,
                signal_buffers=signal_buffers,
            )
            assert image_embeddings is not None

            graph.output(*image_embeddings)
            return graph

    def _build_graph(
        self,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
    ) -> Graph:
        full_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, weights, adapter
        )

        model_config = Qwen3_5Config.initialize_from_config(
            self.pipeline_config, self.huggingface_config
        )
        model_config.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(
                self.huggingface_config
            ),
            state_dict=full_state_dict,
            return_logits=self.return_logits,
            norm_method=self.norm_method,
            attention_bias=self.attention_bias,
        )

        # finalize() reads tie_word_embeddings from the text sub-config,
        # which inherits PretrainedConfig's default of True.  The correct
        # value lives on the top-level config.
        model_config.tie_word_embeddings = getattr(
            self.huggingface_config, "tie_word_embeddings", False
        )
        nn_model = Qwen3_5(model_config)

        graph_inputs = nn_model.input_types(self.kv_params)

        # Log weight loading diagnostics (strict=False silently drops mismatches)
        expected_weights = set(nn_model.raw_state_dict().keys())
        provided_weights = set(full_state_dict.keys())
        missing_keys = expected_weights - provided_weights
        unused_keys = provided_weights - expected_weights
        if missing_keys:
            logger.warning(
                f"Qwen3.5 load_state_dict: {len(missing_keys)} MISSING"
                f" weights (not in checkpoint): {sorted(missing_keys)}"
            )
        if unused_keys:
            logger.info(
                f"Qwen3.5 load_state_dict: {len(unused_keys)} unused"
                f" checkpoint keys: {sorted(list(unused_keys)[:20])}"
                + ("..." if len(unused_keys) > 20 else "")
            )

        nn_model.load_state_dict(
            full_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )

        # Split processed state dict into vision and LM parts.
        # Vision keys keep their "vision_encoder." prefix because the graph
        # resolves weights relative to nn_model (the root), so the registry
        # must match those fully-qualified paths.
        processed = nn_model.state_dict()
        vision_prefix = "vision_encoder."
        self._vision_state_dict = {
            k: v for k, v in processed.items() if k.startswith(vision_prefix)
        } or None
        self.state_dict = {
            k: v
            for k, v in processed.items()
            if not k.startswith(vision_prefix)
        }
        # Keep a reference so _build_vision_graph can access vision_encoder
        self._nn_model = nn_model

        # Save dimensions for state buffer allocation and empty-buffer creation
        self._num_linear_layers = len(nn_model.linear_layer_indices)
        self._conv_dim = nn_model._conv_dim
        self._conv_kernel_size = nn_model._conv_kernel_size
        self._num_v_heads = nn_model._num_v_heads
        self._key_head_dim = nn_model._key_head_dim
        self._value_head_dim = nn_model._value_head_dim
        self._hidden_size = model_config.hidden_size
        self._model_dtype = model_config.dtype

        has_vision = nn_model.vision_encoder is not None
        num_devices = len(self.devices)
        num_linear_layers = self._num_linear_layers
        # Vision adds 2 extra inputs: image_embeddings, image_token_indices
        vision_input_count = 2 if has_vision else 0

        with Graph(
            "qwen3_5",
            input_types=graph_inputs,
        ) as graph:
            tokens, input_row_offsets, return_n_logits, *variadic_args = (
                graph.inputs
            )

            # Extract signal buffers
            signal_buffers = [v.buffer for v in variadic_args[:num_devices]]

            # Unmarshal KV cache inputs. The trailing slice contains
            # [slot_idx, *conv_pools, *recurrent_pools, *vision_inputs].
            kv_start = num_devices
            slot_idx_count = 1 if num_linear_layers > 0 else 0
            kv_count = (
                len(variadic_args)
                - num_devices
                - slot_idx_count
                - num_linear_layers * 2
                - vision_input_count
            )
            kv_cache_inputs = variadic_args[kv_start : kv_start + kv_count]
            kv_collections = self._unflatten_kv_inputs(kv_cache_inputs)

            # Extract slot_idx + the linear-attention pools (BufferType inputs).
            idx = kv_start + kv_count
            slot_idx_g: TensorValue | None = None
            conv_pools: list[BufferValue] = []
            recurrent_pools: list[BufferValue] = []
            if num_linear_layers > 0:
                slot_idx_g = variadic_args[idx].tensor
                idx += 1
                conv_pools = [
                    variadic_args[idx + i].buffer
                    for i in range(num_linear_layers)
                ]
                idx += num_linear_layers
                recurrent_pools = [
                    variadic_args[idx + i].buffer
                    for i in range(num_linear_layers)
                ]
                idx += num_linear_layers

            # Extract vision inputs (only present for multimodal models)
            image_embeddings_g = None
            image_token_indices_g = None
            if has_vision:
                image_embeddings_g = variadic_args[idx].tensor
                image_token_indices_g = variadic_args[idx + 1].tensor

            assert slot_idx_g is not None, (
                "Qwen3.5 graph requires linear attention layers; got 0"
            )
            outputs = nn_model(
                tokens.tensor,
                kv_collections,
                return_n_logits.tensor,
                input_row_offsets.tensor,
                signal_buffers,
                slot_idx_g,
                conv_pools,
                recurrent_pools,
                image_embeddings_g,
                image_token_indices_g,
            )

            graph.output(*outputs)
            return graph

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, Qwen3_5Inputs)
        assert model_inputs.kv_cache_inputs is not None

        if self.vision_model is not None:
            # Multimodal model: always pass image embeddings to the LM graph.
            # For decode/text-only steps, lm_image_embeddings is already the
            # pre-allocated empty buffer from prepare_next_token_inputs.
            # For prefill steps with images, run the vision encoder and update.
            if model_inputs.has_vision_inputs:
                assert model_inputs.pixel_values is not None
                assert model_inputs.weights is not None
                assert model_inputs.indices is not None
                assert model_inputs.vision_position_ids is not None
                assert model_inputs.max_grid_size is not None
                assert model_inputs.grid_thw is not None
                assert model_inputs.cu_seqlens is not None
                assert model_inputs.max_seqlen is not None
                assert model_inputs.image_token_indices is not None

                vision_outputs = self.vision_model.execute(
                    model_inputs.pixel_values,
                    model_inputs.weights,
                    model_inputs.indices,
                    model_inputs.vision_position_ids,
                    model_inputs.max_grid_size,
                    model_inputs.grid_thw,
                    model_inputs.cu_seqlens,
                    model_inputs.max_seqlen,
                    *self.signal_buffers,
                )
                assert isinstance(vision_outputs[0], Buffer)
                assert self._session is not None
                model_inputs.lm_image_embeddings = cast_tensors_to(
                    [vision_outputs[0]], self._model_dtype, self._session
                )[0]
                # image_token_indices is already set on model_inputs
            elif model_inputs.lm_image_embeddings is None:
                # Text-only or decode step with no pre-allocated buffers (e.g.
                # prefill without images): use the persistent empty placeholders.
                assert self._empty_lm_image_embeddings is not None
                assert self._empty_lm_image_token_indices is not None
                model_inputs.lm_image_embeddings = (
                    self._empty_lm_image_embeddings
                )
                model_inputs.image_token_indices = (
                    self._empty_lm_image_token_indices
                )

        model_outputs = self.model.execute(*model_inputs.buffers)

        # The slot-indexed SSM kernels mutate the conv/recurrent pools in
        # place; the only graph output is the logits.
        logits = model_outputs[0]
        assert isinstance(logits, Buffer)

        return ModelOutputs(
            logits=logits,
            next_token_logits=logits,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> Qwen3_5Inputs:
        # Get base Llama3Inputs from parent
        base_inputs = super().prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

        all_contexts = [ctx for batch in replica_batches for ctx in batch]
        request_ids = [ctx.request_id for ctx in all_contexts]

        assert self._state_cache is not None, (
            "Qwen3.5 always has linear-attention layers; state cache must "
            "be initialised by load_model()"
        )
        assert self._slot_idx_prealloc is not None
        # Per-request state management: claim a slot for each request
        # (idempotent for chunked-prefill continuations, zeroes for new ones).
        for rid in request_ids:
            self._state_cache.claim(rid)
        slot_idx = self._state_cache.slot_idx_for(
            request_ids, self._slot_idx_prealloc
        )
        conv_pools = self._state_cache.conv_pools
        recurrent_pools = self._state_cache.rec_pools

        # Vision inputs (only populated for multimodal models with images)
        pixel_values: Buffer | None = None
        weights: Buffer | None = None
        indices: Buffer | None = None
        vision_position_ids: Buffer | None = None
        max_grid_size: Buffer | None = None
        grid_thw: Buffer | None = None
        cu_seqlens: Buffer | None = None
        max_seqlen: Buffer | None = None
        image_token_indices: Buffer | None = None

        if self.vision_model is not None:
            # Collect vision contexts and gather data for contexts with images
            vision_contexts = [
                ctx
                for ctx in all_contexts
                if isinstance(ctx, Qwen3VLTextAndVisionContext)
            ]
            vision_datas: list[VisionEncodingData] = []
            for ctx in vision_contexts:
                if ctx.needs_vision_encoding:
                    assert ctx.vision_data is not None, (
                        "vision_data must be set when needs_vision_encoding is True"
                    )
                    vision_datas.append(ctx.vision_data)

            # Compute scatter indices for merging image embeddings.
            # Fall back to the pre-allocated empty placeholder when there are no
            # vision contexts (e.g. text-only warmup inputs) so that buffers()
            # always has a non-None image_token_indices alongside lm_image_embeddings.
            if vision_contexts:
                np_indices = compute_multimodal_merge_indices(vision_contexts)
                image_token_indices = Buffer.from_numpy(np_indices).to(
                    self.devices[0]
                )
            else:
                image_token_indices = self._empty_lm_image_token_indices

            if vision_datas:
                pixel_values = Buffer.from_numpy(
                    np.concatenate(
                        [vd.concatenated_pixel_values for vd in vision_datas]
                    ).astype(np.float32)
                ).to(self.devices[0])

                weights = Buffer.from_numpy(
                    np.concatenate(
                        [vd.weights for vd in vision_datas], axis=1
                    ).astype(np.float32)
                ).to(self.devices[0])

                indices = Buffer.from_numpy(
                    np.concatenate([vd.indices for vd in vision_datas], axis=1)
                ).to(self.devices[0])

                vision_position_ids = Buffer.from_numpy(
                    np.concatenate(
                        [vd.vision_position_ids for vd in vision_datas]
                    ).astype(np.int32)
                ).to(self.devices[0])

                grid_thw = Buffer.from_numpy(
                    np.concatenate(
                        [vd.image_grid_thw for vd in vision_datas]
                    ).astype(np.int64)
                ).to(self.devices[0])

                max_grid_size_value = max(
                    vd.max_grid_size.item() for vd in vision_datas
                )
                max_grid_size = Buffer.from_numpy(
                    np.array(max_grid_size_value, dtype=np.int32)
                )

                # cu_seqlens: concatenate with cumulative offset adjustments
                cu_seqlens_list = []
                offset = np.uint32(0)
                for vd in vision_datas:
                    seqlens = vd.cu_seqlens.copy()
                    seqlens[1:] += offset
                    cu_seqlens_list.append(seqlens[1:])
                    offset = seqlens[-1]
                cu_seqlens = Buffer.from_numpy(
                    np.concatenate(
                        [np.array([0], dtype=np.uint32), *cu_seqlens_list]
                    ).astype(np.uint32)
                ).to(self.devices[0])

                max_seqlen_value = max(
                    vd.max_seqlen.item() for vd in vision_datas
                )
                max_seqlen = Buffer.from_numpy(
                    np.array([max_seqlen_value], dtype=np.uint32)
                )

        return Qwen3_5Inputs(
            tokens=base_inputs.tokens,
            input_row_offsets=base_inputs.input_row_offsets,
            signal_buffers=base_inputs.signal_buffers,
            kv_cache_inputs=base_inputs.kv_cache_inputs,
            return_n_logits=base_inputs.return_n_logits,
            slot_idx=slot_idx,
            conv_pools=conv_pools,
            recurrent_pools=recurrent_pools,
            request_ids=request_ids,
            # lm_image_embeddings is set in execute() after running the vision
            # encoder.  Use the pre-allocated empty placeholder so that buffers()
            # always returns the right input count; execute() will overwrite it.
            lm_image_embeddings=self._empty_lm_image_embeddings,
            image_token_indices=image_token_indices,
            pixel_values=pixel_values,
            vision_position_ids=vision_position_ids,
            weights=weights,
            indices=indices,
            max_grid_size=max_grid_size,
            grid_thw=grid_thw,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Qwen3_5Inputs:
        assert isinstance(prev_model_inputs, Qwen3_5Inputs)
        assert self._input_row_offsets_prealloc is not None
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        # Build slot_idx for this decode step. The pools live on the cache
        # and are mutated in place by the slot-indexed SSM kernels, so the
        # only per-step transfer is the small uint32 slot_idx tensor written
        # into a pre-allocated buffer (no per-step device allocation).
        request_ids = prev_model_inputs.request_ids
        assert self._state_cache is not None
        assert self._slot_idx_prealloc is not None
        assert request_ids is not None
        slot_idx = self._state_cache.slot_idx_for(
            request_ids, self._slot_idx_prealloc
        )
        conv_pools = self._state_cache.conv_pools
        recurrent_pools = self._state_cache.rec_pools

        # For multimodal models, include pre-allocated empty LM vision inputs so
        # that buffers() returns the correct input count for CUDA graph capture.
        lm_image_embeddings = self._empty_lm_image_embeddings
        lm_image_token_indices = self._empty_lm_image_token_indices

        return Qwen3_5Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            signal_buffers=self.signal_buffers,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
            slot_idx=slot_idx,
            conv_pools=conv_pools,
            recurrent_pools=recurrent_pools,
            request_ids=request_ids,
            lm_image_embeddings=lm_image_embeddings,
            # No vision encoder inputs on decode steps
            image_token_indices=lm_image_token_indices,
            pixel_values=None,
            vision_position_ids=None,
            weights=None,
            indices=None,
            max_grid_size=None,
            grid_thw=None,
            cu_seqlens=None,
            max_seqlen=None,
        )

    def release(self, request_id: RequestID) -> None:
        """Release per-request state cache slot when a request completes."""
        if self._state_cache is not None:
            self._state_cache.release(request_id)
