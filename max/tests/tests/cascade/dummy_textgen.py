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
"""Dummy text-generation components for local cascade examples and tests."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from max.experimental.cascade import (
    CascadePipeline,
    ChatMessages,
    GenerateRequest,
    TextGenInterface,
    Worker,
    worker_method,
)

Int32Array = npt.NDArray[np.int32]


class AsciiTokenizer(Worker):
    """Encode and decode ASCII characters for toy text-generation tests."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["cpu"])

    @worker_method()
    async def encode(self, text: str | ChatMessages) -> Int32Array:
        """Convert text or chat messages to ASCII integer tokens."""
        if isinstance(text, list):
            text = "\n".join(
                str(message.get("content", "")) for message in text
            )
        return np.array([ord(char) for char in text], dtype=np.int32)

    @worker_method()
    async def decode(self, token: int) -> str:
        """Convert an integer token back into a single character."""
        return chr(token)

    @worker_method()
    async def decode_streaming(
        self, token_iter: AsyncIterator[int]
    ) -> AsyncIterator[str]:
        """Convert a token stream into a stream of single-character strings."""
        async for token in token_iter:
            yield chr(token)


class Transformer(Worker):
    """Yield a fixed token stream for deterministic tests."""

    def __init__(self) -> None:
        super().__init__(deploy_hints=["gpu"])

    @worker_method()
    async def decode(
        self, req: GenerateRequest, tokens: Int32Array
    ) -> AsyncIterator[int]:
        """Emit ``num_tokens`` copies of the token for ``"A"``."""
        del tokens
        for _ in range(req.num_tokens):
            yield ord("A")


@dataclass
class DummyTextGenPipeline(CascadePipeline, TextGenInterface):
    """Cascade pipeline pairing the dummy tokenizer and transformer workers."""

    tokenizer: AsciiTokenizer
    transformer: Transformer

    def generate(
        self,
        req: GenerateRequest,
        prompt: str | ChatMessages,
    ) -> AsyncIterator[str]:
        """Run text generation from text or OpenAI-style chat messages."""
        tokens = self.tokenizer.encode(prompt)
        gen_tokens = self.transformer.decode(req, tokens)
        return self.tokenizer.decode_streaming(gen_tokens)


async def build_dummy_textgen_pipeline() -> DummyTextGenPipeline:
    """Build the dummy text pipeline."""
    return DummyTextGenPipeline(
        AsciiTokenizer(),
        Transformer(),
    )
