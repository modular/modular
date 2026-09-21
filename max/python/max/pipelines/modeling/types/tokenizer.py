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

"""Defines the :class:`PipelineTokenizer` protocol for language model tokenizers used in MAX pipelines."""

from __future__ import annotations

__all__ = [
    "PipelineTokenizer",
    "PreprocessCacheStatsProbe",
    "PreprocessedImageProbe",
    "TokenizerEncoded",
    "UnboundContextType",
    "VisionPreprocessCacheStats",
]

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from max.pipelines.request import RequestType

if TYPE_CHECKING:
    from .pipeline_variants.text_generation import (
        TextGenerationRequestMessage,
    )

# TODO: Bound this to TextContext, after we've audited the class.
UnboundContextType = TypeVar("UnboundContextType", covariant=True)
TokenizerEncoded = TypeVar("TokenizerEncoded")


@runtime_checkable
class PreprocessedImageProbe(Protocol):
    """Optional tokenizer capability: report already-preprocessed images.

    A tokenizer that caches preprocessed image tensors implements this so the
    API server can skip the pixel decode for an image whose tensor it already
    holds -- nothing downstream reads those pixels. A tokenizer without such a
    cache simply does not implement it, and every image is decoded as before.

    Declared as a protocol rather than read off the tokenizer with
    ``getattr``, because unlike the model-specific media *limits* the route
    also reads (plain data attributes with safe defaults) this is a callable
    carrying a positional-argument contract and a length invariant. The
    protocol is what states them.
    """

    def preprocessed_image_mask(
        self,
        images: list[bytes],
        messages: list[TextGenerationRequestMessage],
    ) -> Sequence[bool]:
        """Which of ``images`` this tokenizer can already serve preprocessed.

        Must not decode, mutate or cache anything: the API server calls it on
        the event loop before deciding what to decode, and the answer is only
        a hint. An entry may be evicted between this call and tokenization, so
        a ``True`` that later misses must remain correct (the image is simply
        decoded then).

        Args:
            images: Raw encoded bytes per image, in request order.
            messages: The request's messages, carrying any per-image sizing
                hints that the cache key folds in.

        Returns:
            One flag per entry in ``images``, in the same order. Returning a
            different length is a contract violation.
        """
        ...


@dataclass(frozen=True)
class VisionPreprocessCacheStats:
    """One media preprocess cache's counters, as of now.

    The three counts are cumulative for the lifetime of the process that
    owns the cache, and the two byte figures are instantaneous. A reader that
    wants rates owns the differencing, and has to survive the counts
    restarting at zero: the cache is rebuilt empty in any process it is
    unpickled into.
    """

    hits: int
    """Lookups the cache answered without preprocessing."""

    misses: int
    """Lookups that had to preprocess."""

    evictions: int
    """Entries dropped to stay inside the byte budget."""

    size_bytes: int
    """Host bytes the cached payloads retain right now."""

    capacity_bytes: int
    """The byte budget, which ``size_bytes`` is evicted down to (``0``:
    caching is disabled)."""


@runtime_checkable
class PreprocessCacheStatsProbe(Protocol):
    """Optional tokenizer capability: report its media preprocess caches.

    An architecture that caches preprocessed media implements this so the
    server can publish the cache's occupancy and eviction pressure, which is
    what separates "this workload has no repeats" from "the cache is
    thrashing at its budget" -- readings a hit rate alone cannot tell apart.

    A separate protocol rather than a defaulted method on
    :class:`PipelineTokenizer`, for the same reason as
    :class:`PreprocessedImageProbe`: ``PipelineTokenizer`` is
    ``runtime_checkable`` and the API server gates request admission on
    ``isinstance`` against it, so a new member there would reject every
    tokenizer that does not define it. This also keeps the absence typed
    rather than a ``getattr`` sniff at a private attribute name, which would
    stop working silently the day an architecture renames its cache.
    """

    def preprocess_cache_stats(
        self,
    ) -> Mapping[str, VisionPreprocessCacheStats]:
        """This tokenizer's preprocess caches, keyed by media kind.

        Must not preprocess, evict or otherwise disturb a cache: the server
        calls it per request on the event loop.

        Returns:
            One entry per cache the tokenizer owns, keyed ``image`` or
            ``video``. An architecture with no video cache simply omits that
            key rather than reporting an empty one.
        """
        ...


@runtime_checkable
class PipelineTokenizer(
    Protocol[UnboundContextType, TokenizerEncoded, RequestType]
):
    """Interface for LLM tokenizers."""

    @property
    def eos_token_ids(self) -> set[int]:
        """The full set of token ids that end generation for this model.

        The tokenizer's declared EOS plus any additional terminators the
        model ends its turn with (for example, chat turn-end tokens from the
        model's generation config).
        """
        ...

    @property
    def expects_content_wrapping(self) -> bool:
        """If ``True``, this tokenizer expects messages to be wrapped as a dict.

        Text messages are formatted as:

        .. code-block:: json

            {
              "role": "user",
              "content": [{ "type": "text", "text": "text content" }]
            }

        instead of:

        .. code-block:: json

            { "role": "user", "content": "text_content" }

        NOTE: Multimodal messages omit the ``content`` property.
        Both :obj:`image_urls` and :obj:`image` content parts are converted to:

        .. code-block:: json

            { "type": "image" }

        Their content is provided as byte arrays through the top-level property
        on the request object, that is, :obj:`RequestType.images`.
        """
        ...

    async def new_context(self, request: RequestType) -> UnboundContextType:
        """Creates a new context from a request object.

        This is sent to the worker process once and then cached locally.

        Args:
            request: Incoming request.

        Returns:
            Initialized context.
        """
        ...

    async def encode(
        self, prompt: str, add_special_tokens: bool
    ) -> TokenizerEncoded:
        """Encodes text prompts as tokens.

        Args:
            prompt: Un-encoded prompt text.
            add_special_tokens: Whether to add special tokens (for example, BOS).

        Raises:
            ValueError: If the prompt exceeds the configured maximum length.
        """
        ...

    async def decode(self, encoded: TokenizerEncoded, **kwargs) -> str:
        """Decodes response tokens to text.

        Args:
            encoded: Encoded response tokens.
            **kwargs: Additional decoder options (for example, ``skip_special_tokens``).

        Returns:
            Un-encoded response text.
        """
        ...
