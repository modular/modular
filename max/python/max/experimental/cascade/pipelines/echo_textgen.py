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
"""Echo text-generation pipeline: real tokenizer, no model compute.

Pairs the real :class:`MAXTokenizer` with an :class:`EchoTransformer` that
replays the prompt tokens back instead of running a model. It exercises the
whole cascade path -- HuggingFace tokenization, cross-pool worker-to-worker
token streaming, and incremental detokenization -- with the GPU model swapped
out, so a benchmark against it measures cascade framework overhead in isolation
(no model forward pass). Select it with an ``echo:`` prefix on
``--models.main.model-path``; the rest of the path is the tokenizer to load.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import Worker, pipeline_method, worker_method
from max.experimental.cascade.interfaces.pipeline import CascadePipeline
from max.experimental.cascade.interfaces.textgen import (
    ChatMessages,
    GenerateRequest,
    TextGenInterface,
)
from max.experimental.cascade.workers.max_tokenizer import MAXTokenizer

Int32Array = npt.NDArray[np.int32]


class EchoTransformer(Worker):
    """Replay prompt tokens as generated tokens, doing no model compute.

    Deployed to the ``gpu`` pool -- the same placement the real
    :class:`~max.experimental.cascade.workers.max_model_worker.MAXModelWorker`
    takes -- so the echoed token stream crosses the same worker/process boundary
    the production pipeline does.
    """

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def decode(
        self, req: GenerateRequest, tokens: Int32Array
    ) -> AsyncIterator[Int32Array]:
        """Stream ``num_tokens`` tokens back, one per chunk, cycling the prompt.

        Yielding one token per chunk mirrors decode-phase streaming (one token
        per scheduler step), so incremental detokenization and per-token
        streaming overhead are exercised exactly as in the real pipeline. The
        prompt tokens are real vocabulary ids, so detokenization does real work.
        """
        prompt = np.asarray(tokens, dtype=np.int32).reshape(-1)
        if prompt.size == 0:
            return
        for i in range(int(req.num_tokens)):
            yield np.array([prompt[i % prompt.size]], dtype=np.int32)


class EchoTextGenPipeline(CascadePipeline, TextGenInterface):
    """Cascade pipeline pairing ``MAXTokenizer`` with ``EchoTransformer``."""

    def __init__(self, model_path: str) -> None:
        """Build the pipeline for the tokenizer at *model_path*.

        Args:
            model_path: Hugging Face repo id (or local path) whose tokenizer the
                worker loads. No model worker is created, so the pipeline config
                is never resolved and no weights are downloaded.
        """
        self.tokenizer = MAXTokenizer(model_path)
        self.transformer = EchoTransformer()

    @pipeline_method
    async def generate_text(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterator[str]:
        """Tokenize, echo, and detokenize a text or chat prompt end to end.

        Identical wiring to
        :class:`~max.experimental.cascade.pipelines.common_textgen.CommonTextGenPipeline`,
        with the model worker replaced by the echo worker: the token stream
        flows worker-to-worker straight into ``decode_stream``.
        """
        tokens = await self.tokenizer.encode(prompt)
        gen_tokens = await self.transformer.decode(req, tokens)
        async for text in await self.tokenizer.decode_stream(gen_tokens):
            yield text
