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
"""Idefics3 pipeline model (ModuleV3).

Handles compilation and execution of both the vision and language models
using the V3 eager API (``F.lazy()`` + ``compile()``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import SafetensorWeights, Weights, WeightsAdapter
from max.nn.kv_cache import KVCacheInputs, KVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.core import TextAndVisionContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from transformers.models.auto.configuration_auto import AutoConfig

from .model_config import Idefics3Config
from .text_model.idefics3_text import Idefics3Language
from .vision_model.idefics3_vision import Idefics3VisionModel
from .weight_adapters import (
    convert_idefics3_language_model_state_dict,
    convert_idefics3_vision_model_state_dict,
)

logger = logging.getLogger("max.pipelines")


def _assert_image_embeddings_invariant(
    image_embeddings: Buffer, image_token_indices: Buffer
) -> None:
    """Validates that image embeddings count matches image token indices count."""
    embed_count = image_embeddings.shape[0]
    indices_count = image_token_indices.shape[0]

    if embed_count != indices_count:
        logger.error(
            f"[CRITICAL] Vision embedding count ({embed_count}) "
            f"!= image token indices count ({indices_count})."
        )

    assert embed_count == indices_count, (
        f"Vision embedding shape mismatch: {embed_count} embeddings "
        f"but {indices_count} indices."
    )


class _VisionStacker:
    """Helper class for efficient parallel stacking of vision patches."""

    def __init__(self, max_workers: int = 24) -> None:
        self._max_workers = max_workers
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def stack(
        self, images: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        out = np.empty((n, *images[0].shape), dtype=images[0].dtype)
        workers = self._max_workers
        step = math.ceil(n / workers)
        slices = [slice(i, min(i + step, n)) for i in range(0, n, step)]

        futures = [
            self._pool.submit(self._copy_block, out, images, sl)
            for sl in slices
        ]
        for f in as_completed(futures):
            f.result()

        return out

    @staticmethod
    def _copy_block(
        out: npt.NDArray[np.floating[Any]],
        images: list[npt.NDArray[np.floating[Any]]],
        sl: slice,
    ) -> None:
        np.copyto(out[sl], np.asarray(images[sl], dtype=images[0].dtype))


@dataclass
class Idefics3Inputs(ModelInputs):
    """Inputs for the Idefics3 model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer

    # Vision inputs
    pixel_values: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None


class Idefics3Model(PipelineModelWithKVCache[TextAndVisionContext]):
    """An Idefics3 pipeline model using the ModuleV3 API."""

    vision_model: Callable[..., Any]
    """The compiled vision model."""

    language_model: Callable[..., Any]
    """The compiled language model."""

    _input_row_offsets_prealloc: Buffer

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_logits,
        )

        self.vision_model, self.language_model = self.load_model()
        self.image_token_id = self.huggingface_config.image_token_id
        self._stacker = _VisionStacker()

    @staticmethod
    def calculate_max_seq_len(
        pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        max_seq_len = pipeline_config.model.max_length
        if max_seq_len:
            return max_seq_len
        text_config = getattr(
            huggingface_config, "text_config", huggingface_config
        )
        return getattr(text_config, "max_position_embeddings", 4096)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Idefics3Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def load_model(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """Compile vision and language models using the V3 API.

        Returns:
            A tuple of (compiled_vision_model, compiled_language_model).
        """
        # Pre-allocation for multi-step execution
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Validate SafetensorWeights requirement
        if not isinstance(self.weights, SafetensorWeights):
            raise ValueError(
                "Idefics3 currently only supports safetensors weights"
            )

        # Get processed state dicts for language and vision models.
        weights_dict = dict(self.weights.items())
        llm_weights_dict = convert_idefics3_language_model_state_dict(
            weights_dict
        )
        vision_weights_dict = convert_idefics3_vision_model_state_dict(
            weights_dict
        )

        # Generate Idefics3 config from HuggingFace config
        idefics3_config = Idefics3Config.initialize(self.pipeline_config)
        idefics3_config.finalize(
            huggingface_config=self.huggingface_config,
            llm_state_dict=llm_weights_dict,
            return_logits=self.return_logits,
        )

        # Compile vision model
        compiled_vision = self._compile_vision_model(
            idefics3_config, vision_weights_dict
        )

        # Compile language model
        compiled_language = self._compile_language_model(
            idefics3_config, llm_weights_dict
        )

        return compiled_vision, compiled_language

    def _compile_vision_model(
        self,
        config: Idefics3Config,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        """Build and compile the vision model using F.lazy()."""
        image_size = config.vision_config.image_size

        pixel_values_type = TensorType(
            DType.bfloat16,
            shape=["batch_size", 3, image_size, image_size],
            device=DeviceRef.GPU(),
        )

        with F.lazy(), default_dtype(config.vision_config.dtype):
            nn_vision = Idefics3VisionModel(config.vision_config)
            nn_vision.to(self.devices[0])

        compiled = nn_vision.compile(pixel_values_type, weights=state_dict)

        return compiled

    def _compile_language_model(
        self,
        config: Idefics3Config,
        state_dict: dict[str, Any],
    ) -> Callable[..., Any]:
        """Build and compile the language model using F.lazy()."""
        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        image_embeddings_type = TensorType(
            self.dtype,
            shape=[
                "num_image_tokens",
                self.huggingface_config.text_config.hidden_size,
            ],
            device=device_ref,
        )
        image_token_indices_type = TensorType(
            DType.int32, shape=["total_image_tokens"], device=device_ref
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        with F.lazy(), default_dtype(config.text_config.dtype):
            nn_language = Idefics3Language(
                config.text_config,
                config.image_token_id,
                self.kv_params,
            )
            nn_language.to(self.devices[0])

        compiled = nn_language.compile(
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            image_embeddings_type,
            image_token_indices_type,
            *flattened_kv_types,
            weights=state_dict,
        )

        return compiled

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
        """Batch pixel_values for vision processing."""
        images = []
        for context in context_batch:
            for img in context.next_images:
                images.append(img.pixel_values)

        if not images:
            return None

        final_images = self._stacker.stack(images)

        return cast_dlpack_to(
            final_images, DType.float32, DType.bfloat16, self.devices[0]
        )

    def _batch_image_token_indices(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
        """Batch image token indices from multiple contexts."""
        indices_and_offsets = []
        batch_offset = 0

        for ctx in context_batch:
            input_ids = ctx.tokens.active
            special_image_token_mask = input_ids == self.image_token_id
            indices = np.where(special_image_token_mask)[0].tolist()

            indices_and_offsets.append([idx + batch_offset for idx in indices])
            batch_offset += ctx.tokens.active_length

        if not indices_and_offsets:
            return None

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )

        return Buffer.from_numpy(np_indices).to(self.devices[0])

    def _create_empty_image_embeddings(self) -> Buffer:
        return Buffer.zeros(
            shape=[0, self.huggingface_config.text_config.hidden_size],
            dtype=self.dtype,
        ).to(self.devices[0])

    def _create_empty_indices(self) -> Buffer:
        return Buffer.zeros(shape=[0], dtype=DType.int32).to(self.devices[0])

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the Idefics3 model."""
        assert model_inputs.kv_cache_inputs is not None, (
            "Idefics3 requires KV cache inputs"
        )
        assert isinstance(model_inputs, Idefics3Inputs)

        # Process vision inputs if present.
        image_embeddings: Buffer
        image_token_indices: Buffer
        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None
            assert model_inputs.image_token_indices is not None

            # Execute vision model: pixel_values -> image_embeddings.
            # V3 compiled model returns Tensor, not Buffer.
            vision_output = self.vision_model(model_inputs.pixel_values)
            image_embeddings = cast(Buffer, vision_output.driver_tensor)
            image_token_indices = model_inputs.image_token_indices

            _assert_image_embeddings_invariant(
                image_embeddings, image_token_indices
            )
        else:
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        # Execute language model.
        language_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
            image_embeddings,
            image_token_indices,
            *model_inputs.kv_cache_inputs.flatten(),
        )

        # Unpack outputs (V3 returns Tensor objects with .driver_tensor).
        if self.return_logits in (ReturnLogits.VARIABLE, ReturnLogits.ALL):
            return ModelOutputs(
                next_token_logits=cast(
                    Buffer, language_outputs[0].driver_tensor
                ),
                logits=cast(Buffer, language_outputs[1].driver_tensor),
                logit_offsets=cast(Buffer, language_outputs[2].driver_tensor),
            )
        else:
            return ModelOutputs(
                next_token_logits=cast(
                    Buffer, language_outputs[0].driver_tensor
                ),
                logits=cast(Buffer, language_outputs[0].driver_tensor),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepare the initial inputs for the first execution pass."""
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]

        # Marshal pixel values first (before overwriting).
        pixel_values = self._prepare_vision_inputs(context_batch)

        # Input row offsets.
        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Ragged token vector.
        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
        input_ids = Buffer.from_numpy(tokens).to(self.devices[0])

        # Image token indices.
        image_token_indices = self._batch_image_token_indices(context_batch)

        return Idefics3Inputs(
            tokens=input_ids,
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            pixel_values=pixel_values,
            kv_cache_inputs=kv_cache_inputs,
            image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Buffer,
        prev_model_inputs: ModelInputs,
    ) -> Idefics3Inputs:
        prev_model_inputs = cast(Idefics3Inputs, prev_model_inputs)
        old_row_offsets = prev_model_inputs.input_row_offsets
        row_offsets_size = old_row_offsets.shape[0]
        next_row_offsets = self._input_row_offsets_prealloc[:row_offsets_size]

        # In multi-step execution, don't re-pass vision inputs.
        return Idefics3Inputs(
            tokens=next_tokens,
            input_row_offsets=next_row_offsets,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            return_n_logits=prev_model_inputs.return_n_logits,
        )
