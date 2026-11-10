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
"""Gemma 4's media layout and image preprocessing graph, for serving.

A checkpoint declares the layout its media is served with and the graph
that preprocesses one image for the vision tower. Both pipeline models,
ModuleV2 and ModuleV3, read them from here.
"""

from __future__ import annotations

from typing import Any

from max._artifact.vision import (
    AspectRatioPreserving,
    ModalityLayout,
    PromptText,
    Span,
    VisionMetadata,
)
from max.dtype import DType
from max.nn.legacy.image_preprocess import image_preprocess_input_types
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.graph_specs import GraphSpec
from transformers import AutoConfig

from ..image_processor import Gemma4ImageProcessor
from ..model_config import video_token_id
from ..processing_utils import load_processor_config
from ..video_processor import Gemma4VideoProcessor
from .preprocess_graph import Gemma4ImagePreprocess


def gemma4_vision_metadata(
    huggingface_config: AutoConfig,
    image_processor: Gemma4ImageProcessor,
    video_processor: Gemma4VideoProcessor,
) -> VisionMetadata | None:
    """Returns the media layout a Gemma 4 checkpoint is served with.

    Returns ``None`` for a text-only checkpoint.

    Args:
        huggingface_config: The checkpoint's config.
        image_processor: The processor the tokenizer sizes images with.
        video_processor: The processor the tokenizer sizes video frames with.
    """
    vision = getattr(huggingface_config, "vision_config", None)
    if vision is None:
        return None

    def layout(token_id: int, max_soft_tokens: int) -> ModalityLayout:
        pooling = vision.pooling_kernel_size
        # A video frame's run uses the image markers around its own
        # placeholder.
        return ModalityLayout(
            patch_size=vision.patch_size,
            merge_size=pooling,
            strategy=AspectRatioPreserving(max_soft_tokens * pooling**2),
            span=Span(
                token_id,
                huggingface_config.boi_token_id,
                huggingface_config.eoi_token_id,
            ),
        )

    return VisionMetadata.from_layouts(
        image=layout(
            huggingface_config.image_token_id, image_processor.max_soft_tokens
        ),
        video=layout(
            video_token_id(huggingface_config), video_processor.max_soft_tokens
        ),
        # MAX declares no byte caps for this family; 64 MiB is the serving
        # engine's own video cap.
        max_image_bytes=0,
        max_video_bytes=64 << 20,
        # The tokenizer expands a clip as space-joined "<timestamp> <run>"
        # units.
        prompt=PromptText(video_unit_separator=" ", video_label_suffix=" "),
    )


def _processor_config(pipeline_config: PipelineConfig) -> dict[str, Any]:
    """Returns the checkpoint's ``processor_config.json``."""
    path = pipeline_config.model
    return load_processor_config(
        path.model_path, revision=path.huggingface_model_revision
    )


def gemma4_checkpoint_vision_metadata(
    pipeline_config: PipelineConfig,
) -> VisionMetadata | None:
    """Returns the media layout a pipeline's Gemma 4 checkpoint is served with.

    Returns ``None`` for a text-only checkpoint.
    """
    huggingface_config = pipeline_config.model.huggingface_config
    assert huggingface_config is not None
    processor_config = _processor_config(pipeline_config)
    return gemma4_vision_metadata(
        huggingface_config,
        Gemma4ImageProcessor(**processor_config.get("image_processor", {})),
        Gemma4VideoProcessor(**processor_config.get("video_processor", {})),
    )


def gemma4_vision_preprocess_spec(
    pipeline_config: PipelineConfig, dtype: DType
) -> GraphSpec:
    """Returns the graph that preprocesses one image for the vision tower.

    It does what the tokenizer's image processor and the vision input packing
    do, for the checkpoint's processor config.

    Args:
        pipeline_config: The pipeline whose checkpoint names the processor
            config.
        dtype: The dtype of the pixel values the tower takes.
    """
    processor = Gemma4ImageProcessor(
        **_processor_config(pipeline_config).get("image_processor", {})
    )
    return GraphSpec(
        Gemma4ImagePreprocess(processor, dtype),
        "gemma4_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )
