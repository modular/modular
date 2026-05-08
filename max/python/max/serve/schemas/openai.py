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

"""OpenAI-compatible request/response schemas for MAX Serve.

Response models are the official Pydantic types from the ``openai`` SDK,
with small subclasses to expose MAX-only response extensions (currently the
``reasoning`` text emitted by reasoning models). Subclassing rather than
relying on pydantic's ``extra='allow'`` keeps the surface explicit so a
typo in field handling code is a static error rather than a silent extra.

Request models are derived from the SDK's ``TypedDict`` "params" types via
``create_model_from_typeddict``, then subclassed to add MAX-only sampling /
routing extensions and to give a few fields stricter pydantic shapes
(messages, tools, response_format, tool_choice). They use ``extra='forbid'``
to match OpenAI's behavior on unknown request fields - misspelled or
unsupported fields surface as 4xx errors instead of being silently dropped.
"""

from __future__ import annotations

import collections.abc
from typing import Any, Literal, get_args, get_origin, get_type_hints

from openai.types import (
    CompletionUsage as CompletionUsage,
    CreateEmbeddingResponse as CreateEmbeddingResponse,
    Embedding as Embedding,
    Model as Model,
)
from openai.types.chat import (
    ChatCompletion as _OpenAIChatCompletion,
    ChatCompletionChunk as _OpenAIChatCompletionChunk,
    ChatCompletionMessage as _OpenAIChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall as ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob as ChatCompletionTokenLogprob,
)
from openai.types.chat.chat_completion import (
    Choice as _OpenAIChatCompletionChoice,
    ChoiceLogprobs as ChatCompletionLogprobs,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as _OpenAIChatCompletionStreamChoice,
    ChoiceDelta as _OpenAIChoiceDelta,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as ChatCompletionMessageToolCallFunction,
)
from openai.types.chat.chat_completion_token_logprob import (
    TopLogprob as TopLogprob,
)
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase as _OpenAIChatCompletionParams,
)
from openai.types.completion import Completion as CreateCompletionResponse
from openai.types.completion_choice import (
    CompletionChoice as CompletionResponseChoice,
    Logprobs as CompletionLogprobs,
)
from openai.types.completion_create_params import (
    CompletionCreateParamsBase as _OpenAITextCompletionParams,
)
from openai.types.completion_usage import (
    PromptTokensDetails as PromptTokensDetails,
)
from openai.types.embedding_create_params import (
    EmbeddingCreateParams as _OpenAIEmbeddingParams,
)
from pydantic import BaseModel, ConfigDict, Field, create_model

# ---------------------------------------------------------------------------
# Response models.
#
# We subclass the OpenAI SDK pydantic types only to declare the MAX-specific
# ``reasoning`` field (emitted by reasoning models). OpenAI's official chat
# completion shapes don't have this field today (it lives on the newer
# Responses API); other inference servers expose an analogous
# ``reasoning_content``. Clients that don't know about ``reasoning`` will
# either accept it as an OpenAI ``extra='allow'`` field or drop it, so we
# remain a strict superset of the OpenAI wire format.
# ---------------------------------------------------------------------------


class ChatCompletionResponseMessage(_OpenAIChatCompletionMessage):
    """OpenAI assistant message extended with MAX ``reasoning`` text."""

    reasoning: str | None = None


class ChatCompletionStreamResponseDelta(_OpenAIChoiceDelta):
    """OpenAI stream delta extended with MAX ``reasoning`` text."""

    reasoning: str | None = None


class ChatCompletionResponseChoice(_OpenAIChatCompletionChoice):
    """Non-streaming chat completion choice using the MAX-extended message."""

    message: ChatCompletionResponseMessage


class ChatCompletionStreamResponseChoice(_OpenAIChatCompletionStreamChoice):
    """Streaming chat completion choice using the MAX-extended delta."""

    delta: ChatCompletionStreamResponseDelta


class CreateChatCompletionResponse(_OpenAIChatCompletion):
    """Chat completion response using MAX-extended choices."""

    # ``list`` is invariant in pydantic field overrides; mypy needs the ignore
    # but pydantic accepts the narrowed element type at runtime.
    choices: list[ChatCompletionResponseChoice]  # type: ignore[assignment]


class CreateChatCompletionStreamResponse(_OpenAIChatCompletionChunk):
    """Streaming chat completion response using MAX-extended choices."""

    choices: list[ChatCompletionStreamResponseChoice]  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Other response shapes.
# ---------------------------------------------------------------------------


class ListModelsResponse(BaseModel):
    object: Literal["list"]
    data: list[Model]


class Error(BaseModel):
    code: str
    message: str
    param: str
    type: str


class ErrorResponse(BaseModel):
    error: Error


# ---------------------------------------------------------------------------
# Request schemas.
#
# The OpenAI SDK ships request shapes as ``TypedDict`` "params" types that
# can't be used directly as FastAPI request bodies. We project them into
# pydantic models here so we get one source of truth for OpenAI's request
# fields - new SDK fields automatically flow through when we bump the
# pinned ``openai`` version.
#
# All request models are configured with ``extra='forbid'`` so misspelled
# or unsupported parameters surface as a 4xx validation error, matching
# ``api.openai.com``'s behavior. MAX-only sampling/routing fields are
# layered on via subclasses.
# ---------------------------------------------------------------------------

_FORBID_EXTRA = ConfigDict(
    extra="forbid",
    # OpenAI's TypedDict params reference further TypedDicts (e.g. for
    # message content parts and tool definitions); pydantic accepts them
    # as ``arbitrary_types_allowed``.
    arbitrary_types_allowed=True,
)


def _model_from_typeddict(name: str, td: type) -> type[BaseModel]:
    """Builds a pydantic ``BaseModel`` mirroring an OpenAI ``TypedDict``.

    Normalizes ``Iterable[X]`` to ``list[X]`` so the resulting field is a
    concrete sequence (pydantic stores ``Iterable`` as a one-shot validator
    iterator that breaks subscripting and re-iteration).

    All fields default to ``None`` because OpenAI marks only a few fields
    (e.g. ``model``, ``messages``, ``input``) as ``Required[...]``; we
    re-declare the truly required ones in the subclass below with no
    default.
    """
    fields: dict[str, Any] = {}
    for field_name, annotation in get_type_hints(td).items():
        # Strip Required/NotRequired qualifiers (valid in TypedDict definitions
        # but rejected by Pydantic's create_model on Python 3.10).
        if getattr(get_origin(annotation), "_name", "") in (
            "Required",
            "NotRequired",
        ):
            (annotation,) = get_args(annotation)
        if get_origin(annotation) is collections.abc.Iterable:
            (inner,) = get_args(annotation)
            annotation = list[inner]  # type: ignore[valid-type]
        fields[field_name] = (annotation, None)
    return create_model(name, __config__=_FORBID_EXTRA, **fields)


class _MaxRequestExtensions(BaseModel):
    """MAX-specific request fields shared by chat and text completions.

    These are NOT part of the OpenAI spec; OpenAI clients won't send them
    and OpenAI servers won't accept them. Other open-source inference
    servers (vLLM, sglang, ...) ship similar extensions.
    """

    model_config = _FORBID_EXTRA

    # Sampling parameters beyond the OpenAI spec.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    thinking_temperature: float | None = None

    # Generation control.
    min_tokens: int | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False

    # Routing / cache hints used by disaggregated serving.
    target_endpoint: str | None = None
    dkv_cache_hint: dict[str, Any] | None = None


# ---- Auto-generated request bases from OpenAI's TypedDict params ----------
#
# These pull in every OpenAI request field automatically; bumping the
# ``openai`` SDK version is enough to pick up new ones. Required fields
# from the underlying TypedDicts (``model``, ``messages``, ``input``) are
# re-declared on the subclasses below so they have no default.

_ChatCompletionParamsBase = _model_from_typeddict(
    "_ChatCompletionParamsBase", _OpenAIChatCompletionParams
)
_TextCompletionParamsBase = _model_from_typeddict(
    "_TextCompletionParamsBase", _OpenAITextCompletionParams
)
_EmbeddingParamsBase = _model_from_typeddict(
    "_EmbeddingParamsBase", _OpenAIEmbeddingParams
)


class CreateChatCompletionRequest(
    _MaxRequestExtensions, _ChatCompletionParamsBase  # type: ignore[misc,valid-type]
):
    """OpenAI chat completion request, extended with MAX fields.

    Inherits every OpenAI request field from
    ``CompletionCreateParamsBase``; adds MAX-only sampling/routing fields
    via ``_MaxRequestExtensions``.
    """

    # Required fields - re-declare so they have no default. Each message
    # is an OpenAI ``ChatCompletionMessageParam`` TypedDict at the JSON
    # level; we type as ``dict[str, Any]`` here because pydantic mangles
    # the SDK's ``Iterable[ContentPart]`` typing inside the union (it
    # stores a one-shot ``ValidatorIterator``). The route reads role and
    # content via dict access.
    model: str
    messages: list[dict[str, Any]] = Field(min_length=1)

    # ``stream`` lives on the OpenAI streaming/non-streaming subclasses, not
    # on ``CompletionCreateParamsBase`` - declare it explicitly here.
    stream: bool | None = False

    # MAX-only chat-template extension.
    chat_template_kwargs: dict[str, Any] | None = None

    # Pre-tokenized prompt injected by the orchestrator for KV cache-aware
    # routing. When set, the route uses these tokens directly instead of
    # tokenizing ``messages`` (the orchestrator must use the same
    # HuggingFace tokenizer as MAX so the IDs match the model vocabulary).
    # If both are provided, ``prompt_tokens`` takes precedence.
    prompt_tokens: list[int] | None = None


class CreateCompletionRequest(
    _MaxRequestExtensions, _TextCompletionParamsBase  # type: ignore[misc,valid-type]
):
    """OpenAI legacy text completion request, extended with MAX fields."""

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    stream: bool | None = False


class CreateEmbeddingRequest(_EmbeddingParamsBase):  # type: ignore[misc,valid-type]
    """OpenAI embedding request."""

    model: str
    input: str | list[str] | list[int] | list[list[int]]


# ---------------------------------------------------------------------------
# MAX-only request/response types not part of the OpenAI spec.
# ---------------------------------------------------------------------------


class CreateAudioGenerationRequest(BaseModel):
    """Audio generation request used by ``/v1/audio/speech``.

    Note: this is a MAX-specific shape, not the OpenAI ``/v1/audio/speech``
    schema. We may align with the OpenAI spec in a follow-up.
    """

    model_config = _FORBID_EXTRA

    model: str
    input: str
    audio_prompt_tokens: list[int]
    audio_prompt_transcription: str
    instructions: str | None = None
    response_format: Literal["wav", "mp3", "pcm"] | None = None
    speed: float | None = None
    min_tokens: int = 0


class CreateAudioGenerationResponse(BaseModel):
    audio_data: bytes
    metadata: dict[str, Any]


class LoadLoraRequest(BaseModel):
    """MAX-only request to dynamically load a LoRA adapter."""

    model_config = _FORBID_EXTRA

    lora_name: str
    lora_path: str


class UnloadLoraRequest(BaseModel):
    """MAX-only request to dynamically unload a LoRA adapter."""

    model_config = _FORBID_EXTRA

    lora_name: str
