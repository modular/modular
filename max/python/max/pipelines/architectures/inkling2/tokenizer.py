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
"""Inkling 2 tokenizer: the shared text-and-vision one on our own processor."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np
from max.pipelines.context import TextAndVisionContext
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import TextAndVisionTokenizer, resolve_eos_token_ids
from max.pipelines.lib.tokenizer import (
    ReasoningDelimitersMixin,
    open_image,
    resolve_single_special_token,
    resolve_token_id_bound,
)
from max.pipelines.modeling.types import TextGenerationRequest, TokenIds
from max.support.image import hash_image
from PIL import Image
from transformers import AutoTokenizer

from .batch_vision_inputs import IMAGE_GRID_THW
from .image_processor import to_rgb
from .processor import Inkling2Processor, load_processor_config

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


# Opens the JSON payload of a tool call. The tool parser matches this exact
# string, so it has to survive detokenization.
TOOL_CALL_JSON_MARKER = "<|content_invoke_tool_json|>"


class Inkling2Tokenizer(ReasoningDelimitersMixin, TextAndVisionTokenizer):
    """Tokenizer for Inkling 2.

    The checkpoint does not ship its own ``AutoProcessor``, so
    :class:`Inkling2Processor` takes its place. A request's images reach it
    with transparency already composited as the reference does, because the
    shared tokenizer would otherwise flatten them onto white.

    ``skipped_special_token_ids`` opts the streaming detokenizer into a
    per-id filter so ``<|content_invoke_tool_json|>`` survives decode for the
    tool parser. Only that marker: a request without ``tools`` runs no tool
    parser, so anything else kept here leaks verbatim into
    ``message.content``.
    """

    reasoning_delimiters = ("<|content_thinking|>", "<|end_message|>")

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
        # ``--chat-template`` overrides the checkpoint's; the base class's
        # ``apply_chat_template`` renders it when set.
        self._chat_template = chat_template
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )
        self._token_id_bound = resolve_token_id_bound(
            self.delegate, pipeline_config
        )
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        self.processor = Inkling2Processor(
            self.delegate, load_processor_config(model_path, revision)
        )
        self.vision_token_ids = [self.processor.image_token_id]

        self._resolve_reasoning_delimiters(self.delegate)

        tool_call_json_id = resolve_single_special_token(
            self.delegate, TOOL_CALL_JSON_MARKER
        )
        self.skipped_special_token_ids: set[int] = set(
            self.delegate.all_special_ids
        ) - {tool_call_json_id}

    async def decode(self, encoded: TokenIds, **kwargs: Any) -> str:
        """Decodes tokens, dropping every special id but the tool-call marker.

        ``skip_special_tokens=True`` would drop the marker too, so filter by id
        here and decode with the flag off.
        """
        # Log-probability responses decode one token id (a plain int) and the
        # CLI passes a token list; normalize both to a rank-1 array.
        token_ids = np.atleast_1d(np.asarray(encoded))

        if not kwargs.get("skip_special_tokens", True):
            return await super().decode(token_ids, **kwargs)

        filtered_ids = [
            token_id
            for token_id in token_ids.tolist()
            if token_id not in self.skipped_special_token_ids
        ]
        return await super().decode(
            np.array(filtered_ids, dtype=np.int64),
            **{**kwargs, "skip_special_tokens": False},
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Builds the context and records each image's token grid and hash.

        Raises:
            InputError: If the request carries a video.
        """
        if request.videos:
            raise InputError("Inkling 2 does not support video inputs.")
        grids = []
        if request.images:
            images = [
                to_rgb(open_image(image))
                for image in request.images_for_processing()
            ]
            decoded: list[Image.Image | None] = [*images]
            request = dataclasses.replace(request, decoded_images=decoded)
            grids = [self.processor.image_processor.grid(i) for i in images]
        context = await super().new_context(request)
        if grids:
            context.extra_model_args[IMAGE_GRID_THW] = np.array(
                grids, dtype=np.int64
            )
            # The vision cache needs a hash either way; the encoded bytes tell
            # images apart, and with no resize the size tier is constant.
            for metadata, raw_bytes in zip(
                context.images, request.images, strict=True
            ):
                metadata.image_hash = hash_image(
                    raw_bytes, self.processor.image_processor.token_size
                )
        return context
