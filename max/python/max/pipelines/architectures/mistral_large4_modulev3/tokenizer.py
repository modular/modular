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
"""Mistral Large 4's tokenizer for text and images."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Any

import numpy as np
from max.pipelines.lib import TextAndVisionTokenizer
from max.pipelines.lib.tokenizer import (
    resolve_eos_token_ids,
    resolve_token_id_bound,
)
from max.support.image import hash_image
from transformers import AutoTokenizer
from typing_extensions import override

from .context import MistralLarge4VisionContext
from .processor import MistralLarge4Processor, load_processor_config

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.modeling.types import TextGenerationRequest


class MistralLarge4VisionTokenizer(TextAndVisionTokenizer):
    """Expands each image into rows of image tokens, as Pixtral does.

    The base constructor loads ``AutoProcessor``, whose ``PixtralProcessor``
    needs torch and torchvision, so this replaces that step with
    :class:`MistralLarge4Processor` over the same ``processor_config.json``
    constants. A request may carry any number of images: the chat template
    emits one ``[IMG]`` placeholder per image part and the processor grows
    each into its own row span, so nothing here restricts the count.
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
        del unused_kwargs
        self.model_path = model_path
        self._chat_template = chat_template

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length
        self.processor = MistralLarge4Processor(
            self.delegate, load_processor_config(model_path, revision)
        )
        self._eos_token_ids = resolve_eos_token_ids(
            self.delegate.eos_token_id, pipeline_config
        )
        self._token_id_bound = resolve_token_id_bound(
            self.delegate, pipeline_config
        )
        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.vision_token_ids = [
            self.processor.image_token_id,
            self.processor.image_break_token_id,
        ]

    @override
    async def new_context(
        self, request: TextGenerationRequest
    ) -> MistralLarge4VisionContext:
        """Builds the context and records which tokens take an embedding.

        The image span also holds break and end tokens, which embed as text;
        only the image tokens scatter, so their positions are stored on the
        context and each image records how many rows the encoder emits for
        it. The image hash is set unconditionally: the vision encoder cache
        keys on it whether or not prefix caching is on.
        """
        context = await super().new_context(request)
        token_ids = np.asarray(context.tokens.all)
        is_image_token = token_ids == self.processor.image_token_id
        for image in context.images:
            image.num_embedding_rows = int(
                is_image_token[image.start_idx : image.end_idx].sum()
            )
            if image.image_hash is None:
                image.image_hash = hash_image(image.pixel_values)
        return MistralLarge4VisionContext(
            **{f.name: getattr(context, f.name) for f in fields(context)},
            image_token_indices=np.flatnonzero(is_image_token).astype(np.int32),
        )
