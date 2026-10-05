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
"""Muse Glimmer tokenizer: the shared text-and-vision one on our own processor."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
from max.pipelines.context import TextAndVisionContext
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import TextAndVisionTokenizer
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import (
    open_image,
    resolve_eos_token_ids,
    resolve_single_special_token,
)
from max.pipelines.modeling.types import (
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import hash_image
from transformers import AutoTokenizer

from .batch_vision_inputs import IMAGE_GRID_THW
from .image_processor import smart_resize
from .processor import MuseGlimmerProcessor

# The template opens every channel body with ``<|message|>`` and closes a
# non-final channel, the reasoning one included, with ``<|eom|>``. ``<|eom|>``
# must never stop generation: the answer channel follows it in the same turn.
_MESSAGE_TOKEN = "<|message|>"
_END_MESSAGE_TOKEN = "<|eom|>"

_REASONING_EFFORTS = frozenset({"low", "medium", "high"})

_PROCESSOR_CONFIG = "processor_config.json"


def _load_processor_config(
    model_path: str, revision: str | None
) -> dict[str, Any]:
    local = Path(model_path) / _PROCESSOR_CONFIG
    if not local.is_file():
        local = Path(
            huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=_PROCESSOR_CONFIG,
                revision=revision,
            )
        )
    return json.loads(local.read_text())


def to_reasoning_strength(options: Mapping[str, Any]) -> dict[str, Any]:
    """Maps OpenAI's ``reasoning_effort`` onto the template's
    ``reasoning_strength`` and drops the thinking toggles it does not read.

    ``xhigh`` is only reachable as ``reasoning_strength`` in
    ``chat_template_kwargs``; any other effort falls back to the template's
    default (``high``).
    """
    options = dict(options)
    effort = options.pop("reasoning_effort", None)
    options.pop("enable_thinking", None)
    options.pop("thinking", None)
    if effort in _REASONING_EFFORTS and "reasoning_strength" not in options:
        options["reasoning_strength"] = effort
    return options


class MuseGlimmerTokenizer(TextAndVisionTokenizer):
    """Tokenizer for Muse Glimmer.

    Stops on the ``eos_token_id`` list in ``generation_config.json``, the only
    file that names ``<|eot|>``, and exposes the reasoning-delimiter ids the
    ``ReasoningPipelineTokenizer`` protocol asks for. Accepts images, not
    videos.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        if chat_template is not None:
            self.delegate.chat_template = chat_template
        self.max_length = max_length or self.delegate.model_max_length

        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )

        huggingface_config = pipeline_config.model.huggingface_config
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.processor = MuseGlimmerProcessor(
            self.delegate, _load_processor_config(model_path, revision)
        )
        self.vision_token_ids = [
            huggingface_config.image_token_id,
            huggingface_config.video_token_id,
        ]

        self._reasoning_start_token_id = resolve_single_special_token(
            self.delegate, _MESSAGE_TOKEN
        )
        self._reasoning_end_token_id = resolve_single_special_token(
            self.delegate, _END_MESSAGE_TOKEN
        )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the chat template with ``reasoning_effort`` translated."""
        return super().apply_chat_template(
            messages, tools, **to_reasoning_strength(chat_template_options)
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Builds the context and records each image's patch grid and hash.

        Raises:
            InputError: If the request carries a video.
        """
        if request.videos:
            raise InputError("Muse Glimmer does not support video inputs.")
        context = await super().new_context(request)
        if context.images:
            image_processor = self.processor.image_processor
            side = image_processor.patch_size * image_processor.merge_size
            grids = []
            for image in request.images_for_processing():
                width, height = open_image(image).size
                grid_h, grid_w = smart_resize(
                    height, width, side, image_processor.max_image_tokens
                )
                p = image_processor.patch_size
                grids.append((1, grid_h // p, grid_w // p))
            context.extra_model_args[IMAGE_GRID_THW] = np.array(
                grids, dtype=np.int64
            )
            # The base hashes decoded pixels and only with prefix caching on;
            # the vision cache needs a hash either way, and equal pixel arrays
            # can come from different grids.
            for metadata, raw_bytes in zip(
                context.images, request.images, strict=True
            ):
                metadata.image_hash = hash_image(
                    raw_bytes, image_processor.max_image_tokens
                )
        return context

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<|message|>``."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``<|eom|>``."""
        return self._reasoning_end_token_id
