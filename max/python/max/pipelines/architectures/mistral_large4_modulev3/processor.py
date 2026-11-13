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
"""Mistral Large 4 image preprocessing in NumPy and PIL.

Stands in for transformers' ``PixtralProcessor``, which needs torch and
torchvision even for a PIL image, neither of which the serving image ships.
The sizing, pixels and token layout follow the HF path the goldens
recorded (``README-golden-vision.md``): shrink by ``max(H, W) / 1540`` with
floor when over the longest edge, then round each side *up* to a multiple of
``patch_size * spatial_merge_size`` pixels and never upscale; RGB, bicubic
resize, rescale ``1/255``, CLIP normalization, float32 ``[3, H, W]`` rows.

The pixel path is PIL's bicubic. The HF reference's processor resizes
through torchvision (its ``PixtralProcessor`` imports it), so an image
that needs resizing differs from the reference pixels by up to ~2/255 in
raw units; an image that keeps its size agrees to float32 rounding (the
normalize differs by at most 2 ulp, whichever order the two paths round
in). That fidelity caveat is why the golden-pixel parity gate feeds the
S1 pixel arrays rather than re-preprocessing the PNGs; the Mach engine
path keeps its own bit-exact HF port (S4).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.weights.hf_utils import RepoFileSource, fetch_repo_file
from max.support.math import ceildiv
from PIL import Image
from transformers import PreTrainedTokenizerBase

PROCESSOR_CONFIG_FILE = "processor_config.json"


def load_processor_config(
    model_path: str, revision: str | None = None
) -> dict[str, Any]:
    """Reads ``processor_config.json`` out of the tokenizer's repo.

    The Hub form goes through :func:`fetch_repo_file`, so it inherits MAX's
    offline, cache and token handling rather than calling
    ``hf_hub_download`` directly.

    Args:
        model_path: The tokenizer's local directory or Hub repo id.
        revision: The revision the tokenizer files are pinned to.
    """
    local = Path(model_path) / PROCESSOR_CONFIG_FILE
    if local.is_file():
        return json.loads(local.read_text())
    source: RepoFileSource = SimpleNamespace(
        repo_type="local" if local.parent.is_dir() else "hub",
        repo_id=model_path,
        revision=revision or "main",
        subfolder=None,
        local_path=model_path,
    )
    fetched = fetch_repo_file(source, PROCESSOR_CONFIG_FILE)
    if fetched is None:
        raise FileNotFoundError(
            f"{PROCESSOR_CONFIG_FILE} is not in {model_path}"
            + (f" at {revision}" if revision else "")
        )
    return json.loads(fetched.read_text())


class MistralLarge4Processor:
    """NumPy preprocessing after the reference ``PixtralProcessor``.

    Ids with each image placeholder grown into rows of image tokens, and
    one ``[3, H, W]`` float32 pixel array per image.
    """

    def __init__(
        self,
        delegate: PreTrainedTokenizerBase,
        processor_config: Mapping[str, Any],
    ) -> None:
        """Reads the constants from ``processor_config.json``."""
        self.delegate = delegate
        image_config = processor_config["image_processor"]
        # One image token covers a merged square of patches, and the resize
        # rounds each side up to a whole token.
        self.token_side = (
            processor_config["patch_size"]
            * processor_config["spatial_merge_size"]
        )
        self.longest_edge = int(image_config["size"]["longest_edge"])
        self.resample = Image.Resampling(image_config["resample"])
        self.rescale_factor = float(image_config["rescale_factor"])
        self.image_mean = np.asarray(image_config["image_mean"], np.float32)
        self.image_std = np.asarray(image_config["image_std"], np.float32)
        ids = delegate.convert_tokens_to_ids(
            [
                processor_config["image_token"],
                processor_config["image_break_token"],
                processor_config["image_end_token"],
            ]
        )
        assert isinstance(ids, list)
        (
            self.image_token_id,
            self.image_break_token_id,
            self.image_end_token_id,
        ) = ids

    def apply_chat_template(self, messages: Any, **options: Any) -> str:
        """Renders the conversation with the tokenizer's chat template."""
        templated = self.delegate.apply_chat_template(messages, **options)
        assert isinstance(templated, str)
        return templated

    def _resized_shape(self, height: int, width: int) -> tuple[int, int]:
        """Sizes one image like the HF rule.

        Shrinks the longest side to ``longest_edge`` if it is longer, then
        rounds each side up to a multiple of ``token_side``. No upscale: a
        side already under the longest edge only rounds up to its next token
        multiple, as the HF rule does.
        """
        ratio = max(height / self.longest_edge, width / self.longest_edge)
        if ratio > 1:
            height = math.floor(height / ratio)
            width = math.floor(width / ratio)
        return (
            ceildiv(height, self.token_side) * self.token_side,
            ceildiv(width, self.token_side) * self.token_side,
        )

    def _pixels(self, image: Image.Image) -> npt.NDArray[np.float32]:
        height, width = self._resized_shape(image.height, image.width)
        resized = image.convert("RGB").resize(
            (width, height), resample=self.resample
        )
        pixels = np.asarray(resized, dtype=np.float32) * np.float32(
            self.rescale_factor
        )
        return np.ascontiguousarray(
            ((pixels - self.image_mean) / self.image_std).transpose(2, 0, 1)
        )

    def __call__(
        self,
        *,
        text: str | Sequence[int],
        images: Sequence[Image.Image] | None = None,
        add_special_tokens: bool = True,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        """Encodes ``text`` and expands the placeholders.

        Each image token grows into one row of image tokens per
        ``token_side`` pixels of height, each row closed by a break token
        and the last by an end token.

        Returns:
            ``input_ids`` as ``[ids]`` and ``pixel_values`` as ``[[per-image
            [3, H, W] arrays]]``, batched like a HuggingFace processor's
            output.

        Raises:
            ValueError: If the prompt's placeholder count differs from the
                image count.
        """
        del unused_kwargs
        token_ids = (
            self.delegate.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(text, str)
            else list(text)
        )
        pixels = [self._pixels(image) for image in images or []]
        markers = [
            i for i, t in enumerate(token_ids) if t == self.image_token_id
        ]
        if len(markers) != len(pixels):
            raise ValueError(
                f"prompt carries {len(markers)} image placeholder(s) but "
                f"{len(pixels)} image(s) were provided"
            )
        expanded = list(token_ids)
        # Back to front, so the untouched markers keep their indices.
        for index, image_pixels in reversed(
            list(zip(markers, pixels, strict=True))
        ):
            _, height, width = image_pixels.shape
            row = [self.image_token_id] * (width // self.token_side)
            rows = [row + [self.image_break_token_id]] * (
                height // self.token_side
            )
            run = [token for row in rows for token in row]
            run[-1] = self.image_end_token_id
            expanded[index : index + 1] = run
        return {"input_ids": [expanded], "pixel_values": [pixels]}
