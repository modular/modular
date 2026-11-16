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
"""The ``block_diffusion`` component of a model artifact.

It holds the canvas geometry and denoise schedule a serving engine needs,
beside the ``block_diffusion`` graph the artifact's ``model`` section
records: the constants the graph was exported under, which the engine
cannot recover from the MEF. A model that registers no ``BLOCK_DIFFUSION``
module exports no component.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Self

from .value_types import (
    ArtifactFile,
    ArtifactType,
    JsonValue,
    decode_field,
    expect,
)

BLOCK_DIFFUSION_CONFIG_FILE_NAME = "block_diffusion_config.json"


@dataclass(frozen=True)
class BlockDiffusionMetadata:
    """The contents of a model's ``block_diffusion_config.json``.

    Build one with :meth:`from_config`.

    Args:
        config: The decoded ``block_diffusion_config.json``.
    """

    config: Mapping[str, JsonValue]

    @classmethod
    def from_config(
        cls,
        *,
        canvas_length: int,
        max_denoising_steps: int,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        stability_threshold: int,
        confidence_threshold: float,
        pad_token_id: int,
        top_k: int,
        max_canvas_batch: int,
    ) -> Self:
        """Returns the metadata for a registered ``BLOCK_DIFFUSION`` module.

        Args:
            canvas_length: Tokens denoised per canvas.
            max_denoising_steps: The denoise loop's step bound.
            t_min: The temperature schedule's cold end.
            t_max: The temperature schedule's hot end.
            entropy_bound: The entropy-bound acceptance bound.
            stability_threshold: Consecutive stable steps to stop early.
            confidence_threshold: The mean-entropy bound to stop early.
            pad_token_id: The canvas batch's padding token id.
            top_k: The width of the graph's top-k outputs.
            max_canvas_batch: The most canvases one denoise step takes.
        """
        return cls(
            {
                "canvas_length": canvas_length,
                "max_denoising_steps": max_denoising_steps,
                "t_min": t_min,
                "t_max": t_max,
                "entropy_bound": entropy_bound,
                "stability_threshold": stability_threshold,
                "confidence_threshold": confidence_threshold,
                "pad_token_id": pad_token_id,
                "top_k": top_k,
                "max_canvas_batch": max_canvas_batch,
            }
        )

    def __serialize__(self) -> Mapping[str, ArtifactType]:
        """Returns the config as the ``block_diffusion_config.json`` file."""
        text = json.dumps(dict(self.config), indent=2) + "\n"
        return {"config": ArtifactFile(BLOCK_DIFFUSION_CONFIG_FILE_NAME, text)}

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
