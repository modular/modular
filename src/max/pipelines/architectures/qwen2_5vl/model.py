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

from collections.abc import Sequence
from typing import Optional

import numpy as np
from max.driver import Device, Tensor
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph.weights import Weights, WeightsAdapter
from max.pipelines import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModel,
    SupportedEncoding,
)
from max.pipelines.architectures.qwen2_5vl.nn.data_processing import (
    get_window_index,
    mrope_pos_ids_3d,
)
from max.pipelines.context import TextAndVisionContext
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
    """ image pixel_values stacked in 2D tensor of shape [n_patches, in_channels *
    temporal_patch_size * patch_size * patch_size] """
    pixel_values: Tensor | None

    """ grid of spatial and temporal coordinates of patches in pixel_values
    """
    image_grid_thw: Tensor | None
    """ ids for each patch in pixel_values according to their dims.
    """
    image_rot_pos_ids: Tensor | None
    """ video pixel_values
    """
    pixel_values_videos: Tensor | None
    """ grid of spatial and temporal coordinates of patches in pixel_values_videos
    """
    video_grid_thw: Tensor | None
    """ ids for each patch in pixel_values_videos according to their dims.
    """
    video_rot_pos_ids: Tensor | None
    """ indices of image patches included in windows for WindowAttention
    """
    image_window_index: Tensor | None
    """ cumulative seq_len of windows for WindowAttention on images
    """
    image_cu_window_seqlens: Tensor | None
    """ indices of video patches included in windows for WindowAttention
    """
    video_window_index: Tensor | None
    """ cumulative seq_len of windows for WindowAttention on videos
    """
    video_cu_window_seqlens: Tensor | None

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
        image_window_index: Tensor | None,
        image_cu_window_seqlens: Tensor | None,
        video_window_index: Tensor | None,
        video_cu_window_seqlens: Tensor | None,
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
        self.image_window_index = image_window_index
        self.image_cu_window_seqlens = image_cu_window_seqlens
        self.video_window_index = video_window_index
        self.video_cu_window_seqlens = video_cu_window_seqlens
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
        adapter: Optional[WeightsAdapter] = None,
        return_n_logits: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            huggingface_config,
            encoding,
            devices,
            kv_cache_config,
            weights,
            adapter,
            return_n_logits,
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

    def _generate_pos_ids_and_window_indices(
        self, pixel_values: np.ndarray, grid_thw: np.ndarray
    ):
        position_ids = Tensor.from_numpy(
            mrope_pos_ids_3d(
                grid_thw,
                self.huggingface_config.spatial_merge_size,
            )
        )

        window_index, cu_window_seqlens = get_window_index(
            grid_thw,
            self.huggingface_config.window_size,
            self.huggingface_config.spatial_merge_size,
            self.huggingface_config.patch_size,
            self.huggingface_config.spatial_merge_unit,
        )
        return (
            pixel_values,
            grid_thw,
            position_ids,
            Tensor.from_numpy(window_index),
            Tensor.from_numpy(cu_window_seqlens),
        )

    def _prepare_vision_inputs(
        self,
        context_batch: Sequence[TextAndVisionContext],
        has_images: bool,
        has_videos: bool,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Returns
            image_pixel_values,
            image_grid_thw,
            image_position_ids,
            image_window_index,
            image_cu_window_seqlens,
            pixel_values_videos,
            video_grid_thw,
            video_position_ids,
            video_window_index,
            video_cu_window_seqlens,
        """
        # TODO: Call _generate_pos_ids_and_window_indices on image and video inputs for all contexts in the batch
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
        # TODO: Update this after vision context is updated to check for video pixel values too.
        has_videos = False

        # Prepare vision inputs if applicable.
        (
            pixel_values,
            image_grid_thw,
            img_position_ids,
            image_window_index,
            image_cu_window_seqlens,
            pixel_values_videos,
            video_grid_thw,
            video_position_ids,
            video_window_index,
            video_cu_window_seqlens,
        ) = self._prepare_vision_inputs(
            context_batch, has_images=has_images, has_videos=has_videos
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
            image_window_index=image_window_index,
            image_cu_window_seqlens=image_cu_window_seqlens,
            video_window_index=video_window_index,
            video_cu_window_seqlens=video_cu_window_seqlens,
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
        devices: list[Device],
        huggingface_config: AutoConfig,
    ) -> int:
        raise NotImplementedError
