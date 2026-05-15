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
from max.experimental.sharding import (
    DeviceMesh,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.graph.buffer_utils import cast_dlpack_to
from max.graph.weights import Weights, WeightsAdapter
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
from transformers import AutoConfig

from .model_config import Gemma3ForConditionalGenerationConfig
from .vision_model.gemma3multimodal import (
    Gemma3LanguageModel,
    Gemma3VisionModel,
)
from .weight_adapters import (
    convert_safetensor_language_state_dict,
    convert_safetensor_vision_state_dict,
)

logger = logging.getLogger("max.pipelines")


class _VisionStacker:
    """Helper class for efficient parallel stacking of vision patches."""

    def __init__(self, max_workers: int = 24) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def stack(
        self, images: list[npt.NDArray[np.floating[Any]]]
    ) -> npt.NDArray[np.floating[Any]]:
        n = len(images)
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        out = np.empty((n, *images[0].shape), dtype=images[0].dtype)

        workers = self._pool._max_workers
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
class Gemma3MultiModalModelInputs(ModelInputs):
    """Inputs for the Gemma3 multimodal model (V3)."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    pixel_values: Buffer | None = None
    image_token_indices: Buffer | None = None

    @property
    def has_vision_inputs(self) -> bool:
        return self.pixel_values is not None


class Gemma3MultiModalModelV3(
    PipelineModelWithKVCache[TextAndVisionContext],
):
    """Gemma 3 multimodal pipeline model using the ModuleV3 API."""

    language_model: Callable[..., Any]
    vision_model: Callable[..., Any]

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

        self._stacker = _VisionStacker()
        self.vision_model, self.language_model = self._load_models()

    @classmethod
    def estimate_activation_memory(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        del pipeline_config, huggingface_config
        return 15 * 1024 * 1024 * 1024  # 15 GiB

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        return Gemma3ForConditionalGenerationConfig.calculate_max_seq_len(
            pipeline_config, huggingface_config
        )

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        return Gemma3ForConditionalGenerationConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return Gemma3ForConditionalGenerationConfig.get_num_layers(
            huggingface_config
        )

    def _load_models(self) -> tuple[Callable[..., Any], Callable[..., Any]]:
        """Loads vision and language models using the ModuleV3 API."""
        assert self.pipeline_config.runtime.max_batch_size, (
            "Expected max_batch_size to be set"
        )

        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.pipeline_config.runtime.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        weights_dict = dict(self.weights.items())
        language_weights_dict = convert_safetensor_language_state_dict(
            weights_dict
        )
        vision_weights_dict = convert_safetensor_vision_state_dict(weights_dict)

        raw_state_dict = {k: v.data() for k, v in weights_dict.items()}
        model_config = Gemma3ForConditionalGenerationConfig.initialize(
            self.pipeline_config
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=raw_state_dict,
            return_logits=self.return_logits,
        )
        self.config = model_config

        device_ref = DeviceRef.from_device(self.devices[0])
        n_devices = len(self.devices)

        mesh = DeviceMesh(tuple(self.devices), (n_devices,), ("tp",))

        # ---- Build and compile vision model ----
        with F.lazy():
            vision_nn = Gemma3VisionModel(model_config)
            vision_nn.to(self.devices[0])

        pixel_values_type = TensorType(
            DType.bfloat16,
            shape=[
                "batch_size",
                3,
                model_config.vision_config.image_size,
                model_config.vision_config.image_size,
            ],
            device=device_ref,
        )

        compiled_vision = vision_nn.compile(
            pixel_values_type,
            weights=vision_weights_dict,
        )

        # ---- Build and compile language model ----
        with F.lazy():
            language_nn = Gemma3LanguageModel(model_config, self.kv_params)
            language_nn.to(mesh)

        tokens_type = TensorType(
            DType.int64,
            shape=["total_seq_len"],
            device=device_ref,
        )
        image_embeddings_type = TensorType(
            DType.bfloat16,
            shape=[
                "num_image_tokens",
                model_config.text_config.hidden_size,
            ],
            device=device_ref,
        )
        image_token_indices_type = TensorType(
            DType.int32,
            shape=["total_image_tokens"],
            device=device_ref,
        )
        input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )

        kv_inputs = self.kv_params.get_symbolic_inputs()
        flattened_kv_types = kv_inputs.flatten()

        compiled_language = language_nn.compile(
            tokens_type,
            return_n_logits_type,
            input_row_offsets_type,
            image_embeddings_type,
            image_token_indices_type,
            *flattened_kv_types,
            weights=language_weights_dict,
        )

        return compiled_vision, compiled_language

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        model_inputs = cast(Gemma3MultiModalModelInputs, model_inputs)

        image_embeddings: Buffer
        image_token_indices: Buffer
        if model_inputs.has_vision_inputs:
            assert model_inputs.pixel_values is not None

            vision_output = self.vision_model(model_inputs.pixel_values)
            image_embeddings = cast(Buffer, vision_output[0].driver_tensor)

            assert model_inputs.image_token_indices is not None
            image_token_indices = model_inputs.image_token_indices
        else:
            image_embeddings = self._create_empty_image_embeddings()
            image_token_indices = self._create_empty_indices()

        kv_cache_inputs = model_inputs.kv_cache_inputs
        assert kv_cache_inputs is not None

        model_outputs = self.language_model(
            model_inputs.tokens,
            model_inputs.return_n_logits,
            model_inputs.input_row_offsets,
            image_embeddings,
            image_token_indices,
            *kv_cache_inputs.flatten(),
        )

        def _to_buffer(t: Tensor) -> Buffer:
            """Extracts a Buffer from a potentially distributed Tensor."""
            if t.is_distributed:
                return cast(Buffer, t.local_shards[0].driver_tensor)
            return cast(Buffer, t.driver_tensor)

        if len(model_outputs) == 3:
            return ModelOutputs(
                logits=_to_buffer(model_outputs[1]),
                next_token_logits=_to_buffer(model_outputs[0]),
                logit_offsets=_to_buffer(model_outputs[2]),
            )
        else:
            return ModelOutputs(
                logits=_to_buffer(model_outputs[0]),
                next_token_logits=_to_buffer(model_outputs[0]),
            )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputs[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        if len(replica_batches) > 1:
            raise ValueError("Model does not support DP>1")

        context_batch = replica_batches[0]
        dev = self.devices[0]
        assert kv_cache_inputs is not None

        input_row_offsets = Buffer.from_numpy(
            np.cumsum(
                [0] + [ctx.tokens.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(dev)

        tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])

        pixel_values = self._prepare_vision_inputs(context_batch)
        image_token_indices = self._batch_image_token_indices(context_batch)

        return Gemma3MultiModalModelInputs(
            tokens=Buffer.from_numpy(tokens).to(dev),
            input_row_offsets=input_row_offsets,
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            pixel_values=pixel_values,
            image_token_indices=image_token_indices,
        )

    def prepare_next_token_inputs(
        self, next_tokens: Buffer, prev_model_inputs: ModelInputs
    ) -> ModelInputs:
        prev_model_inputs = cast(Gemma3MultiModalModelInputs, prev_model_inputs)
        row_offsets_size = prev_model_inputs.input_row_offsets.shape[0]

        return Gemma3MultiModalModelInputs(
            tokens=next_tokens,
            input_row_offsets=self._input_row_offsets_prealloc[
                :row_offsets_size
            ],
            return_n_logits=prev_model_inputs.return_n_logits,
            kv_cache_inputs=prev_model_inputs.kv_cache_inputs,
            pixel_values=None,
        )

    def _prepare_vision_inputs(
        self, context_batch: Sequence[TextAndVisionContext]
    ) -> Buffer | None:
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
        indices_and_offsets = []
        batch_offset = 0

        for ctx in context_batch:
            input_ids = ctx.tokens.active

            special_image_token_mask = (
                input_ids == self.config.image_token_index
            )
            indices = np.where(special_image_token_mask)[0]

            if len(indices) > 0:
                indices_and_offsets.append(indices + batch_offset)

            batch_offset += ctx.tokens.active_length

        if not indices_and_offsets:
            return Buffer.zeros(shape=[0], dtype=DType.int32).to(
                self.devices[0]
            )

        np_indices = np.concatenate(indices_and_offsets).astype(
            np.int32, copy=False
        )

        return Buffer.from_numpy(np_indices).to(self.devices[0])

    def _create_empty_image_embeddings(self) -> Buffer:
        return Buffer.zeros(
            shape=[0, self.huggingface_config.text_config.hidden_size],
            dtype=DType.bfloat16,
        ).to(self.devices[0])

    def _create_empty_indices(self) -> Buffer:
        return Buffer.zeros(shape=[0], dtype=DType.int32).to(self.devices[0])
