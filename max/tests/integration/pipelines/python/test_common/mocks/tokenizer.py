# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Utilities for working with mock tokenizers for unit testing"""

import json
import random
import string
from collections.abc import Sequence
from typing import TypeVar, Union

import numpy as np
from max.pipelines import PipelineTokenizer, TokenGeneratorRequest
from max.pipelines.context import InputContext, TextContext

T = TypeVar("T", bound=InputContext)


class MockTextTokenizer(PipelineTokenizer[TextContext, np.ndarray]):
    """Mock tokenizer for use in unit tests."""

    def __init__(
        self,
        model_path: str = "testing/testing",
        max_length: Union[int, None] = None,
        max_new_tokens: Union[int, None] = None,
        seed: int = 42,
        vocab_size: int = 1000,
        **kwargs,
    ):
        self.i = 0
        self.vocab_size = vocab_size
        self.seed = seed
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        random.seed(self.seed)
        chars = list(string.printable)
        random.shuffle(chars)
        self.char_to_int = {}
        self.int_to_char = {}
        for idx, char in enumerate(chars):
            self.char_to_int[char] = idx
            self.int_to_char[idx] = char

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    @property
    def eos(self) -> int:
        return self.vocab_size - 10

    async def new_context(self, request: TokenGeneratorRequest) -> TextContext:
        self.i += 1

        if request.prompt is None and request.messages is None:
            msg = "either prompt or messages must be provided."
            raise ValueError(msg)

        max_new_tokens = None
        if request.max_new_tokens is not None:
            max_new_tokens = request.max_new_tokens
        elif self.max_new_tokens != -1:
            max_new_tokens = self.max_new_tokens

        prompt: Union[str, Sequence[int]]
        if request.prompt is None and request.messages is not None:
            prompt = ".".join(
                [str(message.get("content")) for message in request.messages]
            )
        elif request.prompt is not None:
            assert request.prompt is not None
            prompt = request.prompt
        else:
            msg = "either prompt or messages must be provided."
            raise ValueError(msg)

        if isinstance(prompt, str):
            encoded = await self.encode(prompt)
        else:
            encoded = np.array(prompt)

        if self.max_length:
            if len(encoded) > self.max_length:
                msg = "encoded is greater than the max_length of the tokenizer"
                raise ValueError(msg)

        if request.max_new_tokens:
            max_length = len(encoded) + request.max_new_tokens
        elif self.max_new_tokens:
            max_length = len(encoded) + self.max_new_tokens
        else:
            max_length = None

        json_schema = (
            json.dumps(request.response_format.get("json_schema", None))
            if request.response_format
            else None
        )

        return TextContext(
            cache_seq_id=self.i,
            prompt=prompt,
            max_length=max_length,
            tokens=encoded,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
        )

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> np.ndarray:
        return np.array([self.char_to_int[c] for c in prompt])

    async def decode(self, context: T, encoded: np.ndarray, **kwargs) -> str:
        return "".join([self.int_to_char[c] for c in encoded])
