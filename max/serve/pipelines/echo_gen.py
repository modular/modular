# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union, cast

import numpy as np
from max.pipelines.core import (
    PipelineTokenizer,
    TextContext,
    TextGenerationResponse,
    TextGenerationStatus,
    TextResponse,
    TokenGenerator,
    TokenGeneratorRequest,
    TokenGeneratorRequestMessage,
)


@dataclass
class EchoPipelineTokenizer(
    PipelineTokenizer[TextContext, np.ndarray, TokenGeneratorRequest]
):
    """Echo tokenizer that creates TextContext instances.

    This tokenizer creates TextContext objects for echo generation without requiring
    a real HuggingFace tokenizer. It treats input characters as tokens for simplicity.
    """

    @property
    def eos(self) -> int:
        """Return a dummy EOS token ID."""
        return 0

    @property
    def expects_content_wrapping(self) -> bool:
        """Echo tokenizer doesn't require content wrapping."""
        return False

    async def encode(
        self,
        prompt: Union[str, Sequence[int]],
        add_special_tokens: bool = False,
    ) -> np.ndarray:
        """Encode the prompt into token IDs.

        For simplicity, we convert string characters to their ASCII values as token IDs.
        If prompt is already a sequence of ints, we use it directly.
        """
        if isinstance(prompt, str):
            # Convert string characters to ASCII values as token IDs
            return np.array([ord(char) for char in prompt], dtype=np.int32)
        else:
            # Already a sequence of integers
            return np.array(list(prompt), dtype=np.int32)

    async def decode(
        self, context: TextContext, encoded: np.ndarray, **kwargs
    ) -> str:
        """Decode token IDs back to text.

        Convert ASCII values back to characters.
        """
        if isinstance(encoded, int):
            encoded = np.array([encoded])
        elif encoded.ndim == 0:
            encoded = np.array([encoded.item()])

        # Convert ASCII values back to characters
        try:
            return "".join(chr(int(token_id)) for token_id in encoded)
        except ValueError:
            # Fallback for non-ASCII values
            return "".join(str(int(token_id)) for token_id in encoded)

    async def new_context(self, request: TokenGeneratorRequest) -> TextContext:
        """Creates a new TextContext for echo generation."""

        # Extract prompt from request
        prompt: Union[str, Sequence[int]]
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages is not None:
            prompt = "\n".join(
                [
                    str(message["content"])
                    for message in cast(
                        list[TokenGeneratorRequestMessage], request.messages
                    )
                ]
            )
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # Encode the prompt to get token array
        encoded_prompt = await self.encode(prompt, add_special_tokens=False)

        # Determine max tokens
        max_new_tokens = (
            request.sampling_params.max_new_tokens
            if request.sampling_params.max_new_tokens
            else len(encoded_prompt)
        )
        max_length = len(encoded_prompt) + max_new_tokens

        # Create TextContext manually
        context = TextContext(
            prompt=prompt,
            max_length=max_length,
            tokens=encoded_prompt,
            eos_token_ids={self.eos},  # Set containing the EOS token
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            sampling_params=request.sampling_params,
        )

        # Assign to cache if needed
        context.assign_to_cache(request.index)

        return context


@dataclass
class EchoTokenGenerator(TokenGenerator[TextContext]):
    """Token generator that echoes the prompt tokens in their original order."""

    def __init__(self) -> None:
        # Track the echo index for each request (0-based, counts how many tokens we've echoed)
        self._echo_indices: dict[str, int] = {}

    def next_token(
        self, batch: dict[str, TextContext], num_steps: int = 1
    ) -> dict[str, TextGenerationResponse]:
        responses = {}

        for request_id, context in batch.items():
            if request_id not in responses:
                responses[request_id] = TextGenerationResponse(
                    [], TextGenerationStatus.ACTIVE
                )

            # Initialize echo index if not exists
            if request_id not in self._echo_indices:
                self._echo_indices[request_id] = 0

            for step in range(num_steps):
                echo_idx = self._echo_indices[request_id]
                prompt_tokens = context.prompt_tokens

                # Check if we have more tokens to echo and haven't reached max length
                if (
                    echo_idx < len(prompt_tokens)
                    and context.current_length < context.max_length
                ):
                    # Echo the next token in the original order
                    next_token_id = int(prompt_tokens[echo_idx])

                    # Update the context with the new token
                    context.update(next_token_id)

                    # Add to response
                    responses[request_id].append_token(
                        TextResponse(next_token=next_token_id)
                    )

                    # Move to the next token
                    self._echo_indices[request_id] += 1

                else:
                    # Finished echoing all tokens or reached max length
                    responses[request_id].update_status(
                        TextGenerationStatus.MAXIMUM_LENGTH
                    )
                    # Clean up the echo index
                    if request_id in self._echo_indices:
                        del self._echo_indices[request_id]
                    break  # No more tokens to process for this request

        return responses

    def release(self, context: TextContext) -> None:
        """Clean up any state associated with the context."""
        # Note: We can't easily map context back to request_id here,
        # so we'll rely on the scheduler to clean up properly.
        # In practice, this is fine since the echo indices will be cleaned
        # up when contexts complete normally.
        pass
