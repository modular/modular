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
"""The Pixtral tokenizer of the registered decoder and vision roles."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from max.pipelines.architectures.pixtral.tokenizer import PixtralTokenizer
from max.pipelines.modeling.types import TextGenerationRequest
from max.support.image import hash_image

from .context import PixtralContext

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig


class PixtralModuleV3Tokenizer(PixtralTokenizer):
    """Counts each image's embedding rows and hashes it for the vision cache.

    The vision encoder cache keys an image by its hash whether or not prefix
    caching is on.
    """

    def __init__(
        self, model_path: str, pipeline_config: PipelineConfig, **kwargs: Any
    ) -> None:
        super().__init__(model_path, pipeline_config, **kwargs)
        vision_config = pipeline_config.model.huggingface_config.vision_config
        self._image_size = int(vision_config.image_size)

    async def new_context(
        self, request: TextGenerationRequest
    ) -> PixtralContext:
        """Builds the context, with one embedding row per ``[IMG]`` token."""
        context = await super().new_context(request)
        tokens = context.tokens.all
        image_token_id = self.vision_token_ids[0]
        images = [
            replace(
                image,
                num_embedding_rows=int(
                    np.count_nonzero(
                        tokens[image.start_idx : image.end_idx]
                        == image_token_id
                    )
                ),
                # The encoded bytes and the resize bound fix the patch grid,
                # where equal decoded pixels of different shapes would not.
                image_hash=hash_image(raw_bytes, self._image_size),
            )
            for image, raw_bytes in zip(
                context.images, request.images, strict=True
            )
        ]
        return PixtralContext(
            **{
                field.name: getattr(context, field.name)
                for field in fields(context)
                if field.init and field.name != "images"
            },
            images=images,
        )
