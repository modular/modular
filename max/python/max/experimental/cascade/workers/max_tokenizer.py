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
"""Cascade worker wrapping a HuggingFace tokenizer for MAX model inference."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import numpy.typing as npt
from max.experimental.cascade import ChatMessages, Worker, worker_method
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class MAXTokenizer(Worker):
    """Cascade worker that provides tokenization for a HuggingFace model."""

    def __init__(self, model_path: str) -> None:
        super().__init__(deploy_hints=["cpu"])
        self.model_path = model_path
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[MAXTokenizer]:
        """Load the HuggingFace tokenizer for the worker's lifetime."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        yield self

    @worker_method()
    async def encode(self, prompt: str | ChatMessages) -> npt.NDArray[np.int32]:
        """Tokenize plain text or chat messages into ``int32`` token ids."""
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"
        if isinstance(prompt, list):
            token_ids = self._tokenizer.apply_chat_template(
                prompt,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="np",
            )["input_ids"][0]
        else:
            token_ids = self._tokenizer.encode(prompt, return_tensors="np")[0]
        return np.asarray(token_ids, dtype=np.int32)

    @worker_method()
    async def decode(self, tokens: npt.NDArray[np.int32]) -> str:
        """Decode ``token`` ids back into text."""
        # TODO this is incorrect for partial UTF8 tokens
        assert isinstance(tokens, np.ndarray)
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"
        return self._tokenizer.decode(tokens)
