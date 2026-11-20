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
"""The ``vision`` component of a model artifact.

It holds what a serving engine needs, beside the vision graph, to lay out a
model's images and video and splice them into its prompt.
The component is one file, ``vision_config.json``, of plain layout config that
names no model, so a serving engine and its router size media and build
prompts from the file alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, TypeAlias

from typing_extensions import Self

from .value_types import (
    ArtifactFile,
    ArtifactType,
    JsonValue,
    decode_field,
    expect,
)

VISION_CONFIG_FILE_NAME = "vision_config.json"

# Mach's `multimodal::layout::MAX_IMAGE_PIXELS`: the most pixels an image may
# decode to.
_MAX_IMAGE_PIXELS = 64 * 1024 * 1024

# TODO(DISTINF-523): The resize rules are a closed set, mirrored in
# Mach's `multimodal::ResizeStrategy`; a model that sizes media by another rule
# needs a new variant on both sides.


@dataclass(frozen=True)
class SmartResize:
    """Resizes to multiples of the merged patch within a pixel-area budget.

    Args:
        min_pixels: The smallest resized area.
        max_pixels: The largest resized area.
        min_short_side_pixel: The smallest resized short side, or ``0`` for
            none.
        default_max_long_side_pixel: The long-side cap when a request gives
            none, or ``0`` to bound the area alone.
    """

    rule: ClassVar[str] = "smart_resize"
    min_pixels: int
    max_pixels: int
    min_short_side_pixel: int = 0
    default_max_long_side_pixel: int = 0


@dataclass(frozen=True)
class AspectRatioPreserving:
    """Resizes keeping the aspect ratio, within a patch-count budget.

    Args:
        max_patches: The most patches a resized item may have.
    """

    rule: ClassVar[str] = "aspect_ratio_preserving"
    max_patches: int


@dataclass(frozen=True)
class VideoSmartResize:
    """Resizes a clip's frames so the whole clip fits one pixel budget.

    The budget bounds the clip's frames times their area.

    Args:
        temporal_patch_size: How many frames one patch spans.
        min_pixels: The smallest resized clip volume.
        max_pixels: The largest resized clip volume.
    """

    rule: ClassVar[str] = "video_smart_resize"
    temporal_patch_size: int
    min_pixels: int
    max_pixels: int


@dataclass(frozen=True)
class InklingPatchRows:
    """Grows the long edge without shrinking and pads to whole patch rows.

    Args:
        max_upscale: The largest factor by which either side grows.
        upscale_long_edge: The long edge up to which images grow.
    """

    rule: ClassVar[str] = "inkling_patch_rows"
    max_upscale: int = 2
    upscale_long_edge: int = 2048


@dataclass(frozen=True)
class MuseSmartResize:
    """Picks the merged-patch grid nearest the aspect ratio within a token cap.

    The ideal grid, in merged patches, shrinks to the cap, and of the floor
    and ceiling of each side the grid whose aspect ratio is nearest the
    item's wins. Unlike :class:`SmartResize`, it bounds merged tokens, not
    pixels.

    Args:
        max_tokens: The most merged tokens an item may have.
    """

    rule: ClassVar[str] = "muse_smart_resize"
    max_tokens: int


@dataclass(frozen=True)
class Glm5NextSmartResize:
    """Aligns the canvas up to the merged patch within a decoder-token budget.

    GLM-5.3-Flash's ``smart_resize``: both raw sides align up to
    ``patch_size * merge_size``, a canvas under ``min_tokens`` scales up, and
    one over ``max_tokens`` takes the tallest content height whose aligned
    canvas fits. The budget counts a still image as ``temporal_patch_size``
    frames, each token covering ``temporal_patch_size * (patch * merge)**2``
    pixels. The content is resized to fit inside the canvas, never upscaled
    once it is over the token floor, and the rest of the canvas is zero
    padding.

    Args:
        temporal_patch_size: The frames one temporal patch spans; a still
            image is replicated to fill it.
        min_tokens: The fewest decoder tokens an item may produce.
        max_tokens: The most decoder tokens an item may produce.
        max_entry_tokens: Video only, ``0`` for none: the most decoder tokens
            one temporal patch may span.

    Raises:
        ValueError: If a field is not an integer Mach reads as ``u32``, or
            ``min_tokens`` is over ``max_tokens``.
    """

    rule: ClassVar[str] = "glm5_next_smart_resize"
    temporal_patch_size: int
    min_tokens: int
    max_tokens: int
    max_entry_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "temporal_patch_size",
            "min_tokens",
            "max_tokens",
            "max_entry_tokens",
        ):
            value = getattr(self, name)
            low = 0 if name == "max_entry_tokens" else 1
            if not _is_int_in(value, low, 2**32 - 1):
                raise ValueError(
                    f"{name} {value!r} is not an integer from {low} to 2**32 - 1"
                )
        if self.min_tokens > self.max_tokens:
            raise ValueError(
                f"min_tokens {self.min_tokens} is over max_tokens "
                f"{self.max_tokens}"
            )


@dataclass(frozen=True)
class DeepseekV41Letterbox:
    """Grows small images, then letterboxes them into a span budget.

    An image smaller than ``min_pixels`` grows to that area, and each side
    rounds up to whole patches. If the span then has more than
    ``max_span_tokens`` positions, the target shrinks to the largest
    aspect-preserving size whose span fits. A strip too thin for one merged
    cell across at that cap gets a target one merged cell across, as long as
    the cap allows. The image is fitted into the target with a bicubic
    resize, centered with Python's half-to-even rounding, and padded with
    gray ``(127, 127, 127)``, so it is not stretched.

    The span is not one position per merged patch: each merged row ends
    with a newline position, and a start and an end position frame the
    rows, ``ceil(h / merge) * (ceil(w / merge) + 1) + 2`` positions in all.

    Args:
        min_pixels: The smallest source area; a smaller image grows to it.
        max_span_tokens: The most span positions one image may take, the
            start, newline and end positions included.

    Raises:
        ValueError: If ``min_pixels`` is not an integer from 0 to the 64
            Mi-pixel image cap, or ``max_span_tokens`` is not an integer from
            4, the positions of one merged cell, to ``2**32 - 1``. Mach
            refuses a ``vision_config.json`` with any other value.
    """

    rule: ClassVar[str] = "deepseek_v41_letterbox"
    min_pixels: int
    max_span_tokens: int

    def __post_init__(self) -> None:
        # Mach reads `min_pixels` as u64 and `max_span_tokens` as u32.
        if not _is_int_in(self.min_pixels, 0, _MAX_IMAGE_PIXELS):
            raise ValueError(
                f"min_pixels {self.min_pixels!r} is not an integer from 0 to "
                f"the {_MAX_IMAGE_PIXELS}-pixel image cap"
            )
        if not _is_int_in(self.max_span_tokens, 4, 2**32 - 1):
            raise ValueError(
                f"max_span_tokens {self.max_span_tokens!r} is not an integer "
                "from 4, the span of one merged cell, to 2**32 - 1"
            )


def _is_int_in(value: object, low: int, high: int) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and low <= value <= high
    )


@dataclass(frozen=True)
class Inkling2Blocks:
    """Pads to whole blocks of ``patch_size * merge_size`` without resizing.

    The image keeps its pixels and grows at the bottom and right to the next
    whole block; each block is one embedding row.
    """

    rule: ClassVar[str] = "inkling2_blocks"


ResizeStrategy: TypeAlias = (
    SmartResize
    | AspectRatioPreserving
    | VideoSmartResize
    | InklingPatchRows
    | MuseSmartResize
    | Glm5NextSmartResize
    | DeepseekV41Letterbox
    | Inkling2Blocks
)


@dataclass(frozen=True)
class PerFrame:
    """Lays a clip out as independent frames, each sized like an image."""

    mode: ClassVar[str] = "per_frame"


@dataclass(frozen=True)
class PerClip:
    """Lays a clip out as one unit, sized by its frame count too."""

    mode: ClassVar[str] = "per_clip"


@dataclass(frozen=True)
class PerSegmentedClip:
    """Lays a clip out as segments of temporal patch groups.

    Args:
        temporal_patch_size: How many frames one temporal patch group spans.
        groups_per_segment: The most groups one segment holds.
    """

    mode: ClassVar[str] = "per_segmented_clip"
    temporal_patch_size: int
    groups_per_segment: int


VideoLayout: TypeAlias = PerFrame | PerClip | PerSegmentedClip


@dataclass(frozen=True)
class Span:
    """The token ids of the run spliced into the prompt for one item.

    Args:
        image_token_id: The placeholder repeated once per embedding row.
        begin_marker: The token before the run, or ``None`` for none.
        end_marker: The token after the run, or ``None`` for none.
    """

    image_token_id: int
    begin_marker: int | None = None
    end_marker: int | None = None


@dataclass(frozen=True)
class TemplateMarkers:
    """The markers a chat template writes as text around an item's placeholder.

    They sit outside the spliced run.

    Args:
        begin: The marker before the placeholder.
        end: The marker after the placeholder.
    """

    begin: int
    end: int


@dataclass(frozen=True)
class ModalityLayout:
    """How one modality's items are sized and spliced into the prompt.

    Args:
        patch_size: The vision tower's patch size in pixels.
        merge_size: How many patches per side merge into one embedding row.
        strategy: How an item is resized.
        span: The run spliced into the prompt per item.
        template_markers: The markers the chat template writes around the
            placeholder, or ``None`` when the span carries its own.
        video_layout: How a clip divides into units.
    """

    patch_size: int
    merge_size: int
    strategy: ResizeStrategy
    span: Span
    template_markers: TemplateMarkers | None = None
    video_layout: VideoLayout = field(default_factory=PerFrame)

    def to_json(self) -> dict[str, JsonValue]:
        """Returns the layout as ``vision_config.json`` spells it."""
        return {
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "strategy": {"rule": self.strategy.rule, **asdict(self.strategy)},
            "span": {"static": asdict(self.span)},
            "template_markers": (
                None
                if self.template_markers is None
                else asdict(self.template_markers)
            ),
            "video_layout": {
                "mode": self.video_layout.mode,
                **asdict(self.video_layout),
            },
        }


@dataclass(frozen=True)
class PromptText:
    """The text a router writes around a clip's units and between text parts.

    It is spelled as the model's own processor spells it.

    Args:
        video_unit_separator: What joins consecutive video units.
        video_label_suffix: What sits between a unit's timestamp label and
            its run.
        video_clip_open: What opens a clip, before its first unit.
        video_clip_close: What closes a clip, after its last unit.
        text_part_separator: What joins two adjacent text parts of one
            message.
    """

    video_unit_separator: str = ""
    video_label_suffix: str = ""
    video_clip_open: str = ""
    video_clip_close: str = ""
    text_part_separator: str = ""


@dataclass(frozen=True)
class VisionMetadata:
    """The contents of a model's ``vision_config.json``.

    Build one with :meth:`from_layouts`.

    Args:
        config: The decoded ``vision_config.json``.
    """

    config: Mapping[str, JsonValue]

    @classmethod
    def from_layouts(
        cls,
        *,
        image: ModalityLayout,
        video: ModalityLayout | None,
        max_image_bytes: int,
        max_video_bytes: int,
        prompt: PromptText,
        max_images_per_request: int | None = None,
        max_videos_per_request: int | None = None,
    ) -> Self:
        """Returns the metadata for a model's media layouts.

        Args:
            image: The image layout.
            video: The video layout, or ``None`` for a model that serves no
                video.
            max_image_bytes: The per-image encoded-byte cap, or ``0`` for
                none.
            max_video_bytes: The per-video encoded-byte cap, or ``0`` for
                none.
            prompt: The prompt text around media.
            max_images_per_request: The model's image count cap, or
                ``None`` for none declared. A reader then applies its own
                default: 16 items in Mach and its orchestrator. MAX Serve
                reads the tokenizer's caps instead, and applies none when
                the tokenizer declares none.
            max_videos_per_request: The model's video count cap, or
                ``None`` for none declared, with the same default.
        """
        config: dict[str, JsonValue] = {
            "image": image.to_json(),
            "video": None if video is None else video.to_json(),
            "max_image_bytes": max_image_bytes,
            "max_video_bytes": max_video_bytes,
        }
        if max_images_per_request is not None:
            config["max_images_per_request"] = max_images_per_request
        if max_videos_per_request is not None:
            config["max_videos_per_request"] = max_videos_per_request
        config["prompt"] = asdict(prompt)
        return cls(config)

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the config as the ``vision_config.json`` file."""
        text = json.dumps(dict(self.config), indent=2) + "\n"
        return {"config": ArtifactFile(VISION_CONFIG_FILE_NAME, text)}

    @classmethod
    def __deserialize__(cls, value: Mapping[str, ArtifactType]) -> Self:
        """Rebuilds the metadata written by :meth:`__serialize__`.

        Raises:
            ArtifactError: If the file is missing or not a JSON object.
        """
        return cls(decode_field(value, "config", _read_config))


def _read_config(value: ArtifactType) -> dict[str, JsonValue]:
    content = expect(value, ArtifactFile).content
    if isinstance(content, Path):
        content = content.read_text(encoding="utf-8")
    return expect(json.loads(expect(content, str)), dict)
