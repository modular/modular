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
"""Inkling image preprocessing as a CPU graph, and the image layout it serves."""

import math
from collections import OrderedDict

from max._artifact.vision import (
    InklingPatchRows,
    ModalityLayout,
    PromptText,
    Span,
    VisionMetadata,
)
from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.legacy.graph_groups import VisionRows
from max.nn.legacy.image_preprocess import (
    composite_on_white,
    image_preprocess_input_types,
)
from max.nn.legacy.layer import Module
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.graph_specs import GraphSpec
from transformers import AutoTokenizer

from .model_config import InklingConfig
from .processor import load_processor_config


def _resize_axis(
    pixels: TensorValue, size: TensorValue, axis: int
) -> TensorValue:
    """Applies Pillow's seven-tap, fixed-point Lanczos upscale on one axis."""
    cpu = DeviceRef.CPU()
    source = ops.shape_to_tensor([pixels.shape[axis]])[0]
    extent = "resized_width" if axis == 1 else "resized_height"
    output = ops.range(0, size, out_dim=extent, dtype=DType.int64, device=cpu)
    center = (output.cast(DType.float64) + 0.5) * (
        source.cast(DType.float64) / size.cast(DType.float64)
    )
    taps = ops.range(-3, 4, dtype=DType.int64, device=cpu)
    indices = center.cast(DType.int64).reshape([-1, 1]) + taps.reshape([1, 7])
    distance = indices.cast(DType.float64) - center.reshape([-1, 1]) + 0.5
    angle = distance * math.pi
    safe_angle = ops.where(distance == 0, 1.0, angle)
    coefficients = ops.where(
        distance == 0,
        1.0,
        ops.sin(angle) * ops.sin(angle / 3.0) / (safe_angle * safe_angle / 3.0),
    )
    valid = (indices >= 0) & (indices < source) & (ops.abs(distance) < 3.0)
    coefficients = ops.where(valid, coefficients, 0.0)
    coefficients = coefficients / ops.sum(coefficients, axis=1)
    # Pillow quantizes coefficients to signed 22-bit fixed point and rounds
    # and clips after each pass, including the intermediate horizontal image.
    coefficients = (
        coefficients * (1 << 22)
        + ops.where(
            coefficients < 0, ops.constant(-0.5, DType.float64, cpu), 0.5
        )
    ).cast(DType.int32)
    indices = ops.min(ops.max(indices, 0), source - 1)
    gathered = ops.gather(pixels.cast(DType.int32), indices, axis=axis)
    if axis == 1:
        weighted = gathered * coefficients.reshape([1, output.shape[0], 7, 1])
    else:
        weighted = gathered * coefficients.reshape([output.shape[0], 7, 1, 1])
    summed = ops.squeeze(ops.sum(weighted, axis=axis + 1), axis + 1)
    rounded = (summed + (1 << 21)) // (1 << 22)
    return ops.min(ops.max(rounded, 0), 255).cast(DType.uint8)


class InklingImagePreprocess(Module):
    """Composites on white, upscales with Lanczos and pads patches with -1.

    The patches take ``dtype``, the one the vision tower reads.
    """

    def __init__(self, patch_size: int, dtype: DType) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.dtype = dtype

    def __call__(
        self, image: TensorValue, size: TensorValue
    ) -> OrderedDict[str, object]:
        """Returns one image's named vision inputs, matching the image processor."""
        cpu = DeviceRef.CPU()
        pixels = composite_on_white(image)[0]
        source = ops.shape_to_tensor(pixels.shape[:2])
        long_edge = ops.max(source[0], source[1]).cast(DType.float64)
        scale = ops.min(long_edge * 2.0, ops.max(long_edge, 2048.0)) / long_edge
        resized = ops.floor(source.cast(DType.float64) * scale + 0.5).cast(
            DType.int64
        )
        pixels = _resize_axis(pixels, resized[1], axis=1)
        pixels = _resize_axis(pixels, resized[0], axis=0)
        patch = self.patch_size
        rows = ops.range(
            0,
            size[0] // patch,
            out_dim="patch_rows",
            dtype=DType.int64,
            device=cpu,
        )
        cols = ops.range(
            0,
            size[1] // patch,
            out_dim="patch_cols",
            dtype=DType.int64,
            device=cpu,
        )
        inside = ops.range(0, patch, dtype=DType.int64, device=cpu)
        y = (rows.reshape([-1, 1]) * patch + inside).reshape([-1, 1, patch, 1])
        x = (cols.reshape([-1, 1]) * patch + inside).reshape([1, -1, 1, patch])
        valid = (y < resized[0]) & (x < resized[1])
        indices = ops.min(y, resized[0] - 1) * resized[1] + ops.min(
            x, resized[1] - 1
        )
        values = ops.gather(pixels.reshape([-1, 3]), indices, axis=0).cast(
            self.dtype
        )
        patches = ops.where(ops.unsqueeze(valid, -1), values, -1).reshape(
            [rows.shape[0] * cols.shape[0], patch, patch, 3]
        )
        return OrderedDict(patches=VisionRows(pixels=patches))


def inkling_vision_preprocess_spec(
    config: InklingConfig, dtype: DType
) -> GraphSpec:
    """Returns the CPU graph that packs one image's patches for the tower.

    Args:
        config: The config the vision tower is built with.
        dtype: The dtype of the patches the vision tower takes.

    Returns:
        The image preprocessing graph.
    """
    return GraphSpec(
        InklingImagePreprocess(config.vision_config.patch_size, dtype),
        "inkling_vision_preprocess",
        image_preprocess_input_types(),
        weights={},
    )


def inkling_vision_metadata(
    pipeline_config: PipelineConfig, config: InklingConfig
) -> VisionMetadata:
    """Returns Inkling's patch-row image layout and image placeholder.

    Args:
        pipeline_config: The pipeline whose checkpoint names the processor
            config and the tokenizer.
        config: The config the vision tower is built with.

    Returns:
        The media layout.
    """
    model = pipeline_config.model
    processor = load_processor_config(
        model.model_path, model.huggingface_model_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model.model_path,
        revision=model.huggingface_model_revision,
        trust_remote_code=model.trust_remote_code,
    )
    return VisionMetadata.from_layouts(
        image=ModalityLayout(
            patch_size=config.vision_config.patch_size,
            merge_size=1,
            strategy=InklingPatchRows(),
            span=Span(
                tokenizer.convert_tokens_to_ids(processor["image_token"])
            ),
        ),
        video=None,
        max_image_bytes=0,
        max_video_bytes=0,
        prompt=PromptText(),
    )
