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

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import Weights
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
    TextAndVisionContext,
)
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    mrope_pos_ids_3d,
)
from max.pipelines.kv_cache import KVCacheInputs, KVCacheParams, KVCacheStrategy
from transformers import AutoConfig


class Qwen2_5VLInputs(ModelInputs):
    """A class representing inputs for the Qwen2.5VL model.

    This class encapsulates the input tensors required for the Qwen2.5VL model execution:
    - input_ids: A tensor containing the input token IDs
    - input_row_offsets_or_attn_mask: A tensor containing the offsets for each row in the ragged input sequence,
    or the attention mask for the padded input sequence
    - attention_mask ? Maybe not needed.
    - pixel_values: images
    - image_grid_thw:
    - pixel_values_videos:
    - video_grid_thw:
    """

    # Language model inputs.
    input_id_values: Tensor
    input_row_offsets_or_attn_mask: Tensor
    input_id_max_seq_len: Tensor
    pixel_row_offsets: Tensor

    # Vision model inputs.
    pixel_values: Tensor | None
    image_grid_thw: Tensor | None
    image_rot_pos_ids: Tensor | None
    pixel_values_videos: Tensor | None
    video_grid_thw: Tensor | None
    video_rot_pos_ids: Tensor | None

    def __init__(
        self,
        tokens: Tensor,
        input_row_offsets_or_attn_mask: Tensor,
        input_id_max_seq_len: Tensor,
        pixel_row_offsets: Tensor,
        pixel_values: Tensor | None,
        image_grid_thw: Tensor | None,
        image_rot_pos_ids: Tensor | None,
        pixel_values_videos: Tensor | None,
        video_grid_thw: Tensor | None,
        video_rot_pos_ids: Tensor | None,
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets_or_attn_mask = input_row_offsets_or_attn_mask
        self.input_id_max_seq_len = input_id_max_seq_len
        self.pixel_row_offsets = pixel_row_offsets
        self.pixel_values = pixel_values
        self.image_grid_thw = image_grid_thw
        self.image_rot_pos_ids = image_rot_pos_ids
        self.pixel_values_videos = pixel_values_videos
        self.video_grid_thw = video_grid_thw
        self.video_rot_pos_ids = video_rot_pos_ids
        self.kv_cache_inputs = kv_cache_inputs

    @property
    def input_row_offsets(self) -> Tensor:
        """Gets the row offsets of the ragged input sequence."""
        # TODO(bduke): this should implement a ragged tensor interface.
        return self.input_row_offsets_or_attn_mask


class Qwen2_5VLModel(PipelineModel[TextAndVisionContext]):
    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        huggingface_config: AutoConfig,
        encoding: SupportedEncoding,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
        )
        self.model = self.load_model(session)

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        raise NotImplementedError

    @property
    def vision_max_seq_len(self) -> int:
        """Returns the maximum number of vision tokens."""
        raise NotImplementedError

    def _prepare_vision_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def prepare_initial_token_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        kv_cache_inputs: KVCacheInputs | None = None,
    ) -> Qwen2_5VLInputs:
        if self.kv_cache_config.cache_strategy != KVCacheStrategy.CONTINUOUS:
            msg = "Llama Vision only supports continuous batching"
            raise ValueError(msg)

        def has_image(pixel_values: Sequence[np.ndarray]) -> bool:
            return pixel_values is not None and len(pixel_values) > 0

        has_images = any(has_image(ctx.pixel_values) for ctx in context_batch)
        # TODO: Update this after vision context updates to check for video pixel values too.
        has_videos = any(has_image(ctx.pixel_values) for ctx in context_batch)

        # Prepare vision inputs if applicable.
        pixel_values = None
        image_grid_thw = None
        pixel_values_videos = None
        video_grid_thw = None
        if has_images and has_videos:
            (
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            ) = self._prepare_vision_inputs(context_batch)
            img_position_ids = Tensor.from_numpy(
                mrope_pos_ids_3d(
                    image_grid_thw, self.huggingface_config.spatial_merge_size
                )
            )
            video_position_ids = Tensor.from_numpy(
                mrope_pos_ids_3d(
                    video_grid_thw, self.huggingface_config.spatial_merge_size
                )
            )

        # Input row offset type: ["input_row_offsets_len"], UInt32
        input_id_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0] + [ctx.active_length for ctx in context_batch],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        pixel_row_offsets = Tensor.from_numpy(
            np.cumsum(
                [0]
                + [
                    # Use an input row offset of 0 to mean no image.
                    self.vision_max_seq_len
                    if has_image(ctx.pixel_values)
                    else 0
                    for ctx in context_batch
                ],
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Input Ids: ["total_seq_len"], Int64
        # Create a ragged token vector of length: sum(len(t) for t in tokens).
        tokens = np.concatenate([ctx.next_tokens for ctx in context_batch])
        input_id_values = Tensor.from_numpy(tokens).to(self.devices[0])
        # This lives on host / in the CPU kernel, but is later casted to a scalar on
        # device kernel side. No need for explicit .to(pipeline_config.device) call here.
        input_id_max_seq_len = Tensor.from_numpy(
            np.array(
                [max(ctx.active_length for ctx in context_batch)],
                dtype=np.uint32,
            )
        )

        # Unset the context's pixel values so that subsequent next_token
        # calls reusing the same context won't run the vision encoder.
        for ctx in context_batch:
            ctx.pixel_values = []
            # TODO: Update other attributes too

        return Qwen2_5VLInputs(
            tokens=input_id_values,
            input_row_offsets_or_attn_mask=input_id_row_offsets,
            input_id_max_seq_len=input_id_max_seq_len,
            pixel_row_offsets=pixel_row_offsets,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_rot_pos_ids=img_position_ids,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            video_rot_pos_ids=video_position_ids,
            kv_cache_inputs=kv_cache_inputs,
        )

    def prepare_next_token_inputs(
        self,
        next_tokens: Tensor,
        prev_model_inputs: ModelInputs,
    ) -> Qwen2_5VLInputs:
        raise NotImplementedError

    def load_model(
        self,
        session: InferenceSession,
    ) -> Model:
        raise NotImplementedError

    @classmethod
    def calculate_max_seq_len(
        cls, pipeline_config: PipelineConfig, huggingface_config: AutoConfig
    ) -> int:
        raise NotImplementedError

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        n_devices: int,
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        raise NotImplementedError

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        raise NotImplementedError

    @classmethod
    def estimate_kv_cache_size(
        cls,
        pipeline_config: PipelineConfig,
        available_cache_memory: int,
        devices: List[Device],
        huggingface_config: AutoConfig,
    ) -> int:
        raise NotImplementedError
