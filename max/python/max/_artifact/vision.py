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

# TODO(DISTINF-523): The resize rules are a closed set of three, mirrored in
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


ResizeStrategy: TypeAlias = (
    SmartResize | AspectRatioPreserving | VideoSmartResize
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
        """
        return cls(
            {
                "image": image.to_json(),
                "video": None if video is None else video.to_json(),
                "max_image_bytes": max_image_bytes,
                "max_video_bytes": max_video_bytes,
                "prompt": asdict(prompt),
            }
        )

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
