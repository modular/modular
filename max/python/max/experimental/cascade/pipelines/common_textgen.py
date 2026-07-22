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
"""Common text-generation pipeline backed by real MAX workers.

Pairs a :class:`MAXTokenizer` with a :class:`MAXModelWorker` so a single
:class:`PipelineConfig` drives both tokenization and decoding through the
cascade runtime, rather than tokenizing in the caller and hitting the model
worker directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from max.experimental.cascade.core import pipeline_method
from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.interfaces.textgen import (
    ChatMessages,
    GenerateRequest,
    TextGenInterface,
)
from max.experimental.cascade.workers.max_model_worker import MAXModelWorker
from max.experimental.cascade.workers.max_tokenizer import MAXTokenizer
from max.pipelines.lib.config import PipelineConfig


class CommonTextGenPipeline(CascadePipeline, TextGenInterface):
    """Cascade pipeline pairing ``MAXTokenizer`` and ``MAXModelWorker``."""

    def __init__(self, config: PipelineConfig) -> None:
        """Build the pipeline for *config*.

        Args:
            config: Fully-specified ``PipelineConfig``. Its ``model_path``
                seeds the tokenizer worker and the whole config drives the
                model worker.
        """
        self.tokenizer = MAXTokenizer(config.model.model_path)
        self.model = MAXModelWorker(config)

    @pipeline_method
    async def generate_text(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterator[str]:
        """Tokenize, decode, and detokenize a text or chat prompt end to end."""
        tokens = await self.tokenizer.encode(prompt)
        gen_tokens = await self.model.decode(req, tokens)
        async for chunk in gen_tokens:
            yield await (await self.tokenizer.decode(chunk))
