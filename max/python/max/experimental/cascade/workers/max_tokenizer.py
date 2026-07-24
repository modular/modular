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
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import Worker, worker_method
from max.experimental.cascade.interfaces.textgen import ChatMessages
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

Int32Array = npt.NDArray[np.int32]

# Unicode replacement character; the HF tokenizer emits it when a byte-level
# token sequence ends mid-multibyte-character.
_REPLACEMENT_CHAR = "\ufffd"


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
    async def decode(self, tokens: Int32Array) -> str:
        """Decode ``token`` ids back into text.

        Skips special tokens (e.g. ``<|eot_id|>``) so the streamed text matches
        what max-serve returns rather than leaking control tokens into the
        response.

        This one-shot decode is for non-streaming callers; streaming responses
        should use :meth:`decode_stream`, which handles multibyte characters
        split across chunks.
        """
        assert isinstance(tokens, np.ndarray)
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

    @worker_method()
    async def decode_stream(
        self, token_iter: AsyncIterable[Int32Array]
    ) -> AsyncIterator[str]:
        """Detokenize a stream of token-id chunks into a stream of text.

        This is the streaming detokenization stage: the orchestrator hands it
        the model worker's token stream, and it yields incremental text. When
        deployed, it consumes ``token_iter`` directly from the model worker
        (worker-to-worker), so per-token data never round-trips through the
        orchestrator.

        Uses offset-based incremental decoding (the approach vLLM/TGI use): a
        multibyte character split across chunks decodes to the Unicode
        replacement character, so emission is deferred until the following
        chunk(s) complete it. Special tokens are skipped.
        """
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"

        all_ids: list[int] = []
        # ``prefix_offset``/``read_offset`` bound the window that is re-decoded
        # each step, so cost stays proportional to the un-emitted tail rather
        # than the whole sequence.
        prefix_offset = 0
        read_offset = 0
        async for chunk in token_iter:
            new_ids = np.asarray(chunk, dtype=np.int32).reshape(-1).tolist()
            if not new_ids:
                continue
            all_ids.extend(new_ids)

            prefix_text = self._tokenizer.decode(
                all_ids[prefix_offset:read_offset], skip_special_tokens=True
            )
            new_text = self._tokenizer.decode(
                all_ids[prefix_offset:], skip_special_tokens=True
            )

            if len(new_text) > len(prefix_text) and not new_text.endswith(
                _REPLACEMENT_CHAR
            ):
                prefix_offset = read_offset
                read_offset = len(all_ids)
                yield new_text[len(prefix_text) :]

        # Flush the tail. The loop defers a chunk whose decoded text ends in the
        # replacement char pending completion by a later chunk. If the stream
        # ends while a chunk is still deferred, that tail -- including any
        # complete text preceding the incomplete character -- would otherwise be
        # dropped, so the streamed text would fall short of the one-shot
        # ``decode``. Emit whatever remains past the last emitted offset.
        #
        # ``read_offset`` trails ``len(all_ids)`` only while a chunk is
        # deferred; when the last chunk was emitted they are equal and there is
        # nothing to flush, so skip the extra decodes in that common case.
        if read_offset < len(all_ids):
            final_text = self._tokenizer.decode(
                all_ids[prefix_offset:], skip_special_tokens=True
            )
            prefix_text = self._tokenizer.decode(
                all_ids[prefix_offset:read_offset], skip_special_tokens=True
            )
            if len(final_text) > len(prefix_text):
                yield final_text[len(prefix_text) :]
