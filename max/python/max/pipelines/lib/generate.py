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

"""Fixed-batch generation, the lowest-level entry point to the pipeline API.

This runs a pipeline's forward passes over a batch you choose, with nothing
added on top of the pipeline API. The caller's list of prompts is the batch:
it is never split, re-packed, or chunked, and every
:meth:`GenerateIterator.step` runs exactly one ``pipeline.execute`` in the
caller's thread. That also makes it useful for testing, such as reproducing
and localizing kernel and model accuracy bugs. Serving features (scheduling
and in-flight batching, chunked prefill, prefix caching, LoRA, and structured
output) are tested through ``max serve``. Kernels are not batch-invariant, so
for bit-exact comparisons between MAX runs, use batch size 1 or
``ignore_eos=True``.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import io
import logging
import operator
import shlex
import weakref
from collections.abc import Coroutine, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from types import TracebackType
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt
from max.driver import DeviceSpec
from max.pipelines.context import (
    EOSTracker,
    InputError,
    LogProbabilities,
    PixelContext,
    ProcessorInputs,
    SamplingParams,
    SamplingParamsInput,
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache.kv_connector import KVTransfer
from max.pipelines.modeling.types import (
    InputModality,
    PipelineOutputsDict,
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image

from .config import PipelineConfig
from .interfaces import PipelineModel
from .log_probabilities import _MAX_TOP_LOGPROBS
from .pipeline_variants.block_diffusion_text_generation import (
    BlockDiffusionTextGenerationPipeline,
)
from .registry import PIPELINE_REGISTRY
from .tokenizer import checked_token_ids

__all__ = [
    "BatchPipeline",
    "GenerateIterator",
    "GenerateResult",
    "generate",
]

# A row's Philox key is its seed plus its token count, so rows seeded this
# far apart never share a key while every row is under 2**32 tokens.
_ROW_SEED_STRIDE = 1 << 32
_U64_MASK = (1 << 64) - 1

# The tokenizers pass these to the chat template call themselves, or name
# parameters with them along the way, so a caller's copy would collide.
_TOKENIZER_TEMPLATE_KEYS = frozenset(
    (
        "chat_template",
        "conversation",
        "delegate",
        "fn",
        "messages",
        "prompt",
        "self",
        "tokenize",
        "tools",
    )
)

# The image part entries AutoProcessor reads and the message model drops.
_INLINE_IMAGE_KEYS = ("image", "path", "url", "base64")
_AUTO_PROCESSOR_HINT = (
    "Pass the conversations through AutoProcessor, which moves them into "
    "images."
)

_logger = logging.getLogger("max.pipelines")

_T = TypeVar("_T")

_warned_unwarmed_pipelines: weakref.WeakSet[BatchPipeline] = weakref.WeakSet()

# The modes Pillow writes as PNG. Other modes, such as CMYK, convert to RGBA
# when they carry alpha and to RGB otherwise.
_PNG_MODES = frozenset(
    {"1", "L", "LA", "I", "I;16", "I;16B", "P", "RGB", "RGBA"}
)


class _KVCacheParams(Protocol):
    @property
    def page_size(self) -> int: ...


class _KVCacheClaims(Protocol):
    @property
    def params(self) -> _KVCacheParams: ...

    def claim(self, ctx: TextContext, replica_idx: int = 0) -> None: ...

    def alloc(self, ctx: TextContext) -> KVTransfer: ...

    def release(self, ctx: TextContext) -> None: ...

    def contains(self, ctx: TextContext) -> bool: ...


class _EOSTokenSource(Protocol):
    @property
    def eos_token_ids(self) -> set[int]: ...


# Only text tokenizers know their vocab size. PipelineTokenizer omits it.
@runtime_checkable
class _VocabSizeSource(Protocol):
    @property
    def tokenizer_vocab_size(self) -> int: ...


# Vision tokenizers check ids in new_context, so only text tokenizers expose
# the bound.
@runtime_checkable
class _TokenIdBoundSource(Protocol):
    @property
    def token_id_bound(self) -> int: ...


@runtime_checkable
class _ContextSource(Protocol):
    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextContext: ...


@runtime_checkable
class _OverlapSource(Protocol):
    @property
    def overlap_active(self) -> bool: ...


@runtime_checkable
class _GraphCaptureSource(Protocol):
    @property
    def graph_capture_ready(self) -> bool: ...


class _BatchProcessorRuntime(Protocol):
    @property
    def max_batch_active_tokens(self) -> int: ...


class _BatchProcessor(Protocol):
    @property
    def runtime(self) -> _BatchProcessorRuntime: ...


class _LoRAAdapters(Protocol):
    def is_lora(self, name: str) -> bool: ...

    def is_active_lora(self, name: str) -> bool: ...


class _PipelineModelLimits(Protocol):
    @property
    def max_seq_len(self) -> int: ...

    @property
    def lora_manager(self) -> _LoRAAdapters | None: ...

    @property
    def batch_processor(self) -> _BatchProcessor | None: ...

    @property
    def planned_max_batch_total_tokens(self) -> int | None: ...


class BatchPipeline(Protocol):
    """The pipeline surface :class:`GenerateIterator` drives.

    :class:`~max.pipelines.TextGenerationPipeline` and
    :class:`~max.pipelines.lib.OverlapTextGenerationPipeline` over
    :class:`~max.pipelines.context.TextContext` satisfy it. A pipeline whose
    ``execute`` returns the previous call's results also defines a boolean
    ``overlap_active`` property, read the way serve's schedulers read it. A
    pipeline that can replay captured device graphs defines a boolean
    ``graph_capture_ready`` property, true once its graphs are captured. For
    a ``messages`` batch or an image-capable model, the tokenizer also
    defines ``new_context``, which builds each row. GenerateIterator also
    reads the memory plan's max length, batch token limits and LoRA adapters
    from the private ``_pipeline_model``, the only place they live, so
    another implementation must define it as well.
    """

    @property
    def kv_manager(self) -> _KVCacheClaims:
        """The KV cache manager that owns each row's cache blocks."""
        ...

    @property
    def pipeline_config(self) -> PipelineConfig:
        """The config supplying the DP degree and sampling defaults."""
        ...

    @property
    def tokenizer(self) -> _EOSTokenSource:
        """The tokenizer supplying the model's EOS token ids."""
        ...

    @property
    def max_batch_size(self) -> int:
        """The planned maximum batch size per data-parallel replica."""
        ...

    @property
    def _pipeline_model(self) -> _PipelineModelLimits: ...

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Runs one model step over ``inputs``."""
        ...

    def release(self, request_id: RequestID) -> None:
        """Releases the pipeline's per-request state for ``request_id``."""
        ...


@dataclasses.dataclass(frozen=True)
class GenerateResult:
    """The result of :func:`generate`."""

    sequences: list[list[int]]
    """Each row's prompt followed by its new tokens, in input order, as in
    Hugging Face ``generate``."""

    prompt_lengths: list[int]
    """Each row's prompt length, so ``sequences[i][:prompt_lengths[i]]`` is
    row ``i``'s prompt. A prompt built from ``messages`` has its chat
    template applied and its image placeholders expanded."""

    logits: list[npt.NDArray[np.float32]] | None = None
    """Each row's next-token logits, as :attr:`GenerateIterator.logits`, when
    requested."""

    log_probs: list[LogProbabilities] | None = None
    """Each row's log probabilities, as :attr:`GenerateIterator.log_probs`, when
    requested."""


class _LogitsRecorder:
    """Records one context's next-token logits on each forward."""

    def __init__(self) -> None:
        self.logits: list[npt.NDArray[np.float32]] = []

    def __call__(self, inputs: ProcessorInputs) -> None:
        # The last row predicts the next token whether the model returned
        # one row per sequence or one per prompt position. Copied here,
        # because the next step or graph replay reuses the buffer.
        self.logits.append(
            np.array(inputs.logits[-1, :].to_numpy(), dtype=np.float32)
        )


# A pipeline that is not synchronous hands its pending forward's results to
# whichever batch steps next, so two open batches would trade outputs.
_open_iterators: weakref.WeakKeyDictionary[
    object, weakref.ref[GenerateIterator]
] = weakref.WeakKeyDictionary()


def _is_message(value: object) -> bool:
    return isinstance(value, (Mapping, TextGenerationRequestMessage))


def _int_rows(values: Iterable[Sequence[int]], name: str) -> list[list[int]]:
    rows: list[Sequence[int]] = []
    try:
        rows = list(values)
        return [[operator.index(value) for value in row] for row in rows]
    except TypeError:
        hint = "Wrap one prompt as [ids]."
        if any(
            _is_message(row)
            or (isinstance(row, Sequence) and any(map(_is_message, row)))
            for row in rows
        ):
            hint = "Pass conversations as messages=."
        raise TypeError(
            f"{name} must be a list of integer lists, one per prompt. {hint}"
        ) from None


def _materialize_prompts(
    input_ids: Iterable[Sequence[int]],
    attention_mask: Iterable[Sequence[int]] | None,
) -> list[list[int]]:
    prompts = _int_rows(input_ids, "input_ids")
    if attention_mask is None:
        return prompts
    masks = _int_rows(attention_mask, "attention_mask")
    if [len(mask) for mask in masks] != [len(prompt) for prompt in prompts]:
        raise ValueError(
            "attention_mask must have the same shape as input_ids."
        )
    # GenerateIterator runs prompts unpadded, so it drops the positions a
    # zero masks.
    return [
        [token for token, keep in zip(prompt, mask, strict=True) if keep]
        for prompt, mask in zip(prompts, masks, strict=True)
    ]


@dataclasses.dataclass(frozen=True)
class _Conversation:
    messages: list[TextGenerationRequestMessage]
    images: list[bytes]
    decoded_images: list[Image.Image | None]


def _encode_image(
    image: bytes | Image.Image,
) -> tuple[bytes, Image.Image | None]:
    """Returns an image's encoded bytes and, for a PIL image, the decoded one.

    A PIL image is encoded as PNG, since tokenizers hash the bytes. The
    decoded image returned is the one encoded, so both decode to the same
    pixels.
    """
    if isinstance(image, bytes):
        return image, None
    if image.mode not in _PNG_MODES:
        has_alpha = {"A", "a"} & set(image.getbands())
        image = image.convert("RGBA" if has_alpha else "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), image


def _dict_parts(message: object) -> list[Mapping[object, object]]:
    """Returns a dict message's dict content parts, or none for others."""
    if not isinstance(message, Mapping):
        return []
    content = message.get("content")
    if isinstance(content, str) or not isinstance(content, Sequence):
        return []
    return [part for part in content if isinstance(part, Mapping)]


def _has_inline_image(message: object) -> bool:
    return any(
        part.get("type") == "image"
        and any(part.get(key) is not None for key in _INLINE_IMAGE_KEYS)
        for part in _dict_parts(message)
    )


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _materialize_conversations(
    pipeline_config: PipelineConfig,
    messages: Sequence[
        Sequence[TextGenerationRequestMessage | Mapping[str, object]]
    ],
    images: Sequence[Sequence[bytes | Image.Image]] | None,
    caller: str,
    max_images: int | None,
) -> list[_Conversation]:
    messages = list(messages)
    if any(isinstance(row, str) or _is_message(row) for row in messages):
        raise TypeError(
            "messages must be a list of conversations, one per row. Wrap one "
            "conversation as [messages]."
        )
    messages = [list(row) for row in messages]
    for row, conversation in enumerate(messages):
        if any(
            part.get("type") == "image_url"
            for message in conversation
            for part in _dict_parts(message)
        ):
            raise ValueError(
                f"Row {row} has an image_url part, which {caller} does not "
                f"read. {_AUTO_PROCESSOR_HINT}"
            )
    rows = [
        [
            message
            if isinstance(message, TextGenerationRequestMessage)
            else TextGenerationRequestMessage.model_validate(message)
            for message in row
        ]
        for row in messages
    ]
    if images is None:
        row_images: list[list[bytes | Image.Image]] = [[] for _ in rows]
    else:
        shape_error = TypeError(
            "images must be a list of image lists, one per row, each image "
            "encoded bytes or a PIL image. Wrap one row's images as [images]."
        )
        image_rows = list(images)
        # Hugging Face takes a flat list of PIL images, whose items don't
        # iterate.
        if any(
            isinstance(row, (bytes, bytearray, str))
            or not isinstance(row, Iterable)
            for row in image_rows
        ):
            raise shape_error
        row_images = [list(row) for row in image_rows]
        if not all(
            isinstance(image, (bytes, Image.Image))
            for row in row_images
            for image in row
        ):
            raise shape_error
        if len(row_images) != len(rows):
            raise ValueError(
                f"images has {_plural(len(row_images), 'row')}, but messages "
                f"has {len(rows)}. Pass one list of images per conversation."
            )
    image_parts = [
        sum(message.number_of_images for message in conversation)
        for conversation in rows
    ]
    video_rows = [
        row
        for row, conversation in enumerate(rows)
        if any(message.number_of_videos for message in conversation)
    ]
    if any(image_parts) or video_rows:
        arch = PIPELINE_REGISTRY.architecture_for_config(pipeline_config)
        if arch is None or InputModality.IMAGE not in arch.input_modalities:
            row = next(
                row
                for row, count in enumerate(image_parts)
                if count or row in video_rows
            )
            media = (
                _plural(image_parts[row], "image")
                if image_parts[row]
                else "a video part"
            )
            name = (
                pipeline_config.model.model_path if arch is None else arch.name
            )
            raise ValueError(
                f"Row {row} has {media}, but {name} takes text only."
            )
    if video_rows:
        raise _unsupported(
            pipeline_config,
            "video inputs",
            f"Row {video_rows[0]} has a video part, and {caller} builds text "
            "and image rows only",
            caller=caller,
        )
    if any(image_parts):
        if max_images is not None and max(image_parts) > max_images:
            row = next(
                row
                for row, count in enumerate(image_parts)
                if count > max_images
            )
            raise ValueError(
                f"Row {row} has {_plural(image_parts[row], 'image')}, but this "
                f"pipeline takes at most {_plural(max_images, 'image')} per "
                "row, as max serve does."
            )
    for row, (num_parts, row_entries) in enumerate(
        zip(image_parts, row_images, strict=True)
    ):
        # Checked whatever the image count, since a matching count would
        # otherwise run the row on a different image.
        if any(map(_has_inline_image, messages[row])):
            raise ValueError(
                f'Row {row} has an image part with an "image", "path", "url" '
                f'or "base64" entry, which {caller} does not read. '
                f"{_AUTO_PROCESSOR_HINT}"
            )
        if len(row_entries) != num_parts:
            raise ValueError(
                f"Row {row} has {_plural(num_parts, 'image part')} but "
                f"{_plural(len(row_entries), 'image')}. Pass one image per "
                "image part, in order."
            )
    conversations = []
    for conversation, row_entries in zip(rows, row_images, strict=True):
        encoded = [_encode_image(image) for image in row_entries]
        conversations.append(
            _Conversation(
                conversation,
                [data for data, _ in encoded],
                [decoded for _, decoded in encoded],
            )
        )
    return conversations


def _resolve_sampling_params(
    pipeline: BatchPipeline, requested: SamplingParamsInput
) -> SamplingParams:
    # Unset knobs fall back to the server-level defaults before the
    # generation config, as serve resolves a request's sampling params.
    runtime = pipeline.pipeline_config.runtime
    return SamplingParams.from_input_and_generation_config(
        dataclasses.replace(
            requested,
            temperature=(
                requested.temperature
                if requested.temperature is not None
                else runtime.temperature
            ),
            top_k=(
                requested.top_k
                if requested.top_k is not None
                else runtime.top_k
            ),
        ),
        sampling_params_defaults=pipeline.pipeline_config.model.sampling_params_defaults,
    )


def _run_coroutine(coroutine: Coroutine[object, object, _T]) -> _T:
    """Runs ``coroutine`` to completion from synchronous code.

    It runs on a worker thread with its own loop and the caller's context
    variables, so it works inside a running loop, as in Jupyter, and leaves
    the caller's loop alone.
    """
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(context.run, asyncio.run, coroutine).result()


def _row_token_ids(
    row: int, prompt: Sequence[int], token_id_bound: int | None
) -> npt.NDArray[np.int64]:
    if token_id_bound is None:
        return np.asarray(prompt, dtype=np.int64)
    try:
        return checked_token_ids(prompt, token_id_bound)
    except InputError as error:
        error.args = (f"Row {row}: {error}",)
        raise


async def _new_contexts(
    tokenizer: _ContextSource, requests: Sequence[TextGenerationRequest]
) -> list[TextContext]:
    contexts = []
    for row, request in enumerate(requests):
        try:
            contexts.append(await tokenizer.new_context(request))
        except ValueError as error:
            error.args = (f"Row {row}: {error}",)
            raise
    return contexts


def _row_request(
    row: Sequence[int] | _Conversation,
    sampling_params: SamplingParams,
    chat_template_kwargs: Mapping[str, object] | None,
) -> TextGenerationRequest:
    if isinstance(row, _Conversation):
        return TextGenerationRequest(
            request_id=RequestID(),
            model_name="",
            messages=row.messages,
            images=row.images,
            decoded_images=row.decoded_images,
            sampling_params=sampling_params,
            chat_template_options=(
                None
                if chat_template_kwargs is None
                else dict(chat_template_kwargs)
            ),
        )
    return TextGenerationRequest(
        request_id=RequestID(),
        model_name="",
        prompt=list(row),
        sampling_params=sampling_params,
    )


def _build_contexts(
    pipeline: BatchPipeline,
    rows: Sequence[Sequence[int] | _Conversation],
    *,
    max_new_tokens: int,
    sampling_params: SamplingParams,
    tokenizer_builds_rows: bool,
    chat_template_kwargs: Mapping[str, object] | None,
) -> list[TextContext]:
    """Builds one context per row.

    The pipeline's tokenizer builds every row of a ``messages`` batch, and
    every row on an image-capable model, whose context class carries fields
    ``TextContext`` lacks.
    """
    planned_max_length = pipeline._pipeline_model.max_seq_len
    if sampling_params.min_new_tokens > max_new_tokens:
        raise ValueError(
            f"min_new_tokens={sampling_params.min_new_tokens}, from the "
            "arguments or the generation config, exceeds "
            f"max_new_tokens={max_new_tokens}. Raise max_new_tokens."
        )
    min_new_tokens = max(sampling_params.min_new_tokens, 1)

    def row_max_length(row: int, prompt_length: int) -> int:
        if prompt_length + min_new_tokens > planned_max_length:
            raise ValueError(
                f"Row {row} needs at least {prompt_length + min_new_tokens} "
                f"tokens ({prompt_length} prompt + {min_new_tokens} new), but "
                f"the pipeline's planned max length is {planned_max_length}. "
                "Raise max_length."
            )
        max_length = prompt_length + max_new_tokens
        if sampling_params.ignore_eos and max_length > planned_max_length:
            raise ValueError(
                f"Row {row} needs {max_length} tokens, but the pipeline's "
                f"planned max length is {planned_max_length}, and with "
                "ignore_eos every row runs all max_new_tokens steps. Raise "
                "max_length or lower max_new_tokens."
            )
        return min(planned_max_length, max_length)

    row_sampling_params = [
        dataclasses.replace(
            sampling_params,
            seed=(sampling_params.seed + row * _ROW_SEED_STRIDE) & _U64_MASK,
        )
        for row in range(len(rows))
    ]
    tokenizer = pipeline.tokenizer
    if tokenizer_builds_rows:
        assert isinstance(tokenizer, _ContextSource)
        contexts = _run_coroutine(
            _new_contexts(
                tokenizer,
                [
                    _row_request(row, params, chat_template_kwargs)
                    for row, params in zip(
                        rows, row_sampling_params, strict=True
                    )
                ],
            )
        )
        for row, context in enumerate(contexts):
            # The tokenizer clamps to its own max length, which can exceed
            # the memory plan's.
            context.max_length = row_max_length(row, len(context.tokens))
        return contexts

    # The same set serve's tokenizer builds for a request.
    eos_token_ids: set[int] = set()
    if not sampling_params.ignore_eos:
        eos_token_ids.update(tokenizer.eos_token_ids)
        eos_token_ids.update(sampling_params.stop_token_ids or ())
    vocab_size = (
        tokenizer.tokenizer_vocab_size
        if isinstance(tokenizer, _VocabSizeSource)
        else None
    )
    token_id_bound = (
        tokenizer.token_id_bound
        if isinstance(tokenizer, _TokenIdBoundSource)
        else None
    )
    contexts = []
    for row, prompt in enumerate(rows):
        assert not isinstance(prompt, _Conversation)
        contexts.append(
            TextContext(
                request_id=RequestID(),
                max_length=row_max_length(row, len(prompt)),
                tokens=TokenBuffer(_row_token_ids(row, prompt, token_id_bound)),
                eos_tracker=EOSTracker(eos_token_ids=set(eos_token_ids)),
                sampling_params=row_sampling_params[row],
                vocab_size=vocab_size,
            )
        )
    return contexts


def _pipeline_serve_flags(pipeline_config: PipelineConfig) -> list[str]:
    """Returns the ``max serve`` flags that rebuild this pipeline's model.

    A flag is left out when the CLI default gives the same value on any
    GPU host.
    """
    model = pipeline_config.model
    flags: list[str] = []
    if model.huggingface_weight_repo_id != model.model_path:
        # Only a bare repo id passed as the weight path makes discovery read
        # files from another repo.
        flags += ["--weight-path", model.huggingface_weight_repo_id]
    if model.trust_remote_code:
        flags.append("--trust-remote-code")
    devices = model.device_specs
    # Every GPU host defaults to GPU 0. CPU is spelled out because a GPU
    # host would not pick it.
    if devices and devices != [DeviceSpec.accelerator(0)]:
        if any(device.device_type == "cpu" for device in devices):
            flags += ["--devices", "cpu"]
        else:
            gpu_ids = ",".join(str(device.id) for device in devices)
            flags += ["--devices", f"gpu:{gpu_ids}"]
    if model.quantization_encoding is not None:
        flags += ["--quantization-encoding", model.quantization_encoding]
    if model.data_parallel_degree != 1:
        flags += ["--data-parallel-degree", str(model.data_parallel_degree)]
    # Memory planning never shrinks an explicit --max-length, so repeating
    # an architecture default would change the plan.
    if model.max_length_is_user_provided and model.max_length is not None:
        flags += ["--max-length", str(model.max_length)]
    return flags


def _unsupported(
    pipeline_config: PipelineConfig,
    feature: str,
    reason: str,
    serve_flags: Sequence[str] = (),
    *,
    caller: str = "generate()",
    note: str | None = None,
) -> ValueError:
    """Returns the error for a feature ``caller`` leaves to ``max serve``.

    The message names ``caller`` and ``feature``, gives ``reason``, and ends
    with a ``max serve`` command that rebuilds this pipeline with
    ``serve_flags`` added, then ``note`` if given.
    """
    main = pipeline_config.models.get("main")
    model_flags = (
        [
            "--model-path",
            main.model_path,
            *_pipeline_serve_flags(pipeline_config),
        ]
        if main is not None
        # A diffusion pipeline has only components, all from one repo.
        else [
            "--model-path",
            next(iter(pipeline_config.models.values())).model_path,
        ]
    )
    command = shlex.join(["max", "serve", *model_flags, *serve_flags])
    message = (
        f"{caller} does not support {feature}. {reason}.\n"
        f"{caller} runs the pipeline directly, outside max serve. "
        f"Test {feature} with this max serve command instead.\n  {command}"
    )
    if note is not None:
        message += f"\n{note}"
    return ValueError(message)


def _holds_video(context_type: type[PixelContext]) -> bool:
    # Only a video model's context carries a frame count.
    return any(
        field.name == "num_frames" for field in dataclasses.fields(context_type)
    )


def _check_supported(
    pipeline_config: PipelineConfig, caller: str = "generate()"
) -> type[TextContext]:
    """Raises if ``caller`` cannot run the pipeline's model.

    The error names ``caller``, the class the user built. A feature only
    serve runs gets the ``max serve`` command to use instead.

    Returns:
        The context class the model's rows use.
    """
    context_type = PIPELINE_REGISTRY.retrieve_context_type(pipeline_config)
    arch = PIPELINE_REGISTRY.architecture_for_config(pipeline_config)
    if issubclass(context_type, PixelContext) and not _holds_video(
        context_type
    ):
        raise ValueError(
            f"{caller} generates text, and the model uses "
            f"{context_type.__name__}. Use generate_images() for image "
            "generation models."
        )
    if not issubclass(context_type, TextContext):
        raise _unsupported(
            pipeline_config,
            f"{context_type.__name__} models"
            if arch is None
            else arch.task.value.replace("_", " "),
            f"The model uses {context_type.__name__}, and {caller} generates "
            "text",
            caller=caller,
        )
    if (
        arch is not None
        and arch.pipeline_cls is not None
        and issubclass(arch.pipeline_cls, BlockDiffusionTextGenerationPipeline)
    ):
        raise _unsupported(
            pipeline_config,
            "block diffusion models",
            f"{arch.name} decodes a block of tokens per step with its own "
            "sampling, so it ignores temperature, top_k and seed and returns "
            "no logits",
            caller=caller,
        )
    lora = pipeline_config.lora
    if lora is not None and lora.enable_lora:
        raise _lora_unsupported(
            pipeline_config,
            f"{caller} runs only the base model and never activates an adapter",
            caller=caller,
        )
    return context_type


def _lora_unsupported(
    pipeline_config: PipelineConfig,
    reason: str,
    feature: str = "LoRA adapters",
    local_note: str | None = None,
    caller: str = "generate()",
) -> ValueError:
    """Returns the LoRA error, with serve flags that load the same adapters.

    ``local_note``, if given, follows the serve note to say how to run the
    batch in-process instead.
    """
    lora = pipeline_config.lora
    assert lora is not None
    flags = ["--enable-lora", "--no-enable-prefix-caching"]
    for path in lora.lora_paths:
        flags += ["--lora-paths", path]
    flags += [
        "--max-lora-rank",
        str(lora.max_lora_rank),
        "--max-num-loras",
        str(lora.max_num_loras),
    ]
    if lora.lora_paths:
        # The same naming rule serve's LoRAQueue applies.
        names = ", ".join(
            f'"{path.partition("=")[0]}"' for path in lora.lora_paths
        )
        note = (
            'Then send requests with "model": "<adapter name>". An '
            "adapter's name is the part of its --lora-paths entry before "
            f'"=", or the whole entry if it has no "=" (names here: '
            f"{names})."
        )
    else:
        note = (
            "Load an adapter with POST /v1/load_lora_adapter, then send "
            'requests with "model": "<lora_name>".'
        )
    if local_note is not None:
        note += f"\n{local_note}"
    return _unsupported(
        pipeline_config, feature, reason, flags, caller=caller, note=note
    )


def _check_adapters_active(
    pipeline: BatchPipeline, contexts: Sequence[TextContext], caller: str
) -> None:
    """Raises if a context names a loaded LoRA adapter that is not active.

    Serve's scheduler activates a request's adapter. GenerateIterator does
    not, so the row would run the base weights without an error.
    """
    lora_manager = pipeline._pipeline_model.lora_manager
    if lora_manager is None:
        return
    for row, context in enumerate(contexts):
        name = context.model_name
        if lora_manager.is_lora(name) and not lora_manager.is_active_lora(name):
            raise _lora_unsupported(
                pipeline.pipeline_config,
                f'Row {row} names the adapter "{name}", which is loaded but '
                f"not active, and {caller} never activates an adapter",
                feature="inactive LoRA adapters",
                caller=caller,
                local_note=(
                    "To run it here, activate it first with "
                    "max.pipelines.lora.get_lora_manager(pipeline)"
                    f'.activate_adapter("{name}").'
                ),
            )


def _check_graph_capture(
    pipeline: BatchPipeline, dp_degree: int, caller: str = "generate()"
) -> None:
    """Checks GenerateIterator against the pipeline's graph capture state.

    Raises if replay needs replica padding, and warns once per pipeline if
    capture is on but the pipeline is not warmed up, so a caller that opens
    many batches logs it once. Reads whether the pipeline captured its
    graphs, not the config flag, so a pipeline warmed up by hand gets the
    same check.
    """
    pipeline_config = pipeline.pipeline_config
    ready = (
        isinstance(pipeline, _GraphCaptureSource)
        and pipeline.graph_capture_ready
    )
    if not ready:
        if (
            pipeline_config.runtime.device_graph_capture
            and pipeline not in _warned_unwarmed_pipelines
        ):
            _warned_unwarmed_pipelines.add(pipeline)
            _logger.warning(
                "Device graph capture is on, but warmup_graph_capture() has "
                "not run on this pipeline, so every step runs eagerly. To "
                "replay captured graphs, call pipeline.warmup_graph_capture() "
                "first, or load the model with "
                "AutoModel(..., device_graph_capture=True)."
            )
        return
    if dp_degree > 1:
        raise _unsupported(
            pipeline_config,
            "device graph capture with data parallelism",
            "Replay needs every replica to step the same number of rows. "
            f"Serve pads the replicas to keep them equal, and {caller} does "
            "not pad",
            ["--device-graph-capture"],
            caller=caller,
            note=(
                "To run this here, load the model without device graph capture."
            ),
        )


def _check_penalties_enabled(
    pipeline_config: PipelineConfig, penalties: Mapping[str, float]
) -> None:
    """Raises if the caller set ``penalties`` the pipeline's sampler skips.

    ``penalties`` holds the penalty arguments the caller set to a non-default
    value.
    """
    if penalties and not pipeline_config.sampling.enable_penalties:
        requested = ", ".join(
            f"{name}={value}" for name, value in penalties.items()
        )
        raise ValueError(
            "This pipeline was built with enable_penalties=False, so its "
            f"sampler would ignore {requested}. Rebuild it with "
            "sampling=SamplingConfig(enable_penalties=True) (in max serve: "
            "--enable-penalties)."
        )


def _check_output_options(
    pipeline: BatchPipeline,
    synchronous: bool,
    output_logits: bool,
    output_logprobs: int,
    caller: str,
) -> None:
    """Raises if the pipeline cannot return the requested outputs."""
    if not 0 <= output_logprobs <= _MAX_TOP_LOGPROBS:
        raise ValueError(
            f"output_logprobs must be in [0, {_MAX_TOP_LOGPROBS}], was "
            f"{output_logprobs}."
        )
    if (
        output_logits or output_logprobs
    ) and pipeline.pipeline_config.speculative is not None:
        raise ValueError(
            "output_logits and output_logprobs need one logits row per new "
            "token, and with speculative decoding a step can add several "
            "tokens to a row."
        )
    if output_logits and not synchronous:
        advice = (
            "Load the AutoModel without enable_overlap_scheduler=True."
            if caller == "AutoModel"
            else "Use AutoModel, which turns the overlap scheduler off."
        )
        raise ValueError(
            "output_logits needs a synchronous pipeline, but this pipeline "
            "has overlap active, so each step returns the tokens sampled "
            f"from the previous step's logits. {advice}"
        )
    # The pipeline checks this only after the rows have claimed KV.
    if output_logprobs and isinstance(pipeline, _OverlapSource):
        raise ValueError(
            "OverlapTextGenerationPipeline rejects log probabilities even "
            "with overlap off. Pass output_logits=True and take log_softmax "
            "over the logits, or use AutoModel without device graph capture."
        )
    model_type = type(pipeline._pipeline_model)
    # The pipeline would warn and return no log probabilities after prefill.
    if output_logprobs and (
        getattr(model_type, "compute_log_probabilities", None)
        is PipelineModel.compute_log_probabilities
    ):
        raise ValueError(
            f"{model_type.__name__} does not compute log probabilities. Pass "
            "output_logits=True and take log_softmax over the logits."
        )


def _align_up(value: int, alignment: int) -> int:
    return -(-value // alignment) * alignment


def _check_batch_fits(
    pipeline: BatchPipeline,
    prompt_lengths: Sequence[int],
    dp_degree: int,
    caller: str = "generate()",
) -> None:
    """Raises if the batch needs more rows or tokens than the plan allows."""
    pipeline_config = pipeline.pipeline_config
    max_rows = pipeline.max_batch_size * dp_degree
    if len(prompt_lengths) > max_rows:
        raise _unsupported(
            pipeline_config,
            "scheduling across batches",
            f"{len(prompt_lengths)} prompts exceed the pipeline's max batch "
            f"size of {max_rows}"
            + (
                f" ({pipeline.max_batch_size} per replica x {dp_degree} "
                "replicas)"
                if dp_degree > 1
                else ""
            )
            + f", and {caller} never splits its prompts",
            ["--max-batch-size", str(pipeline.max_batch_size)],
            caller=caller,
            note="To run this batch here, raise max_batch_size.",
        )
    model = pipeline._pipeline_model
    batch_processor = model.batch_processor
    # The scheduler chunks each replica's prefill to fit its own share of
    # this budget. GenerateIterator does not chunk, and an overflow would
    # otherwise fail deep in graph input staging.
    token_budget = (
        None
        if batch_processor is None
        else batch_processor.runtime.max_batch_active_tokens // dp_degree
    )
    total_tokens_cap = model.planned_max_batch_total_tokens
    page_size = pipeline.kv_manager.params.page_size
    for replica_idx in range(dp_degree):
        lengths = prompt_lengths[replica_idx::dp_degree]
        if token_budget is not None and sum(lengths) > token_budget:
            raise _unsupported(
                pipeline_config,
                "chunked prefill",
                f"Replica {replica_idx}'s prompts total {sum(lengths)} "
                "tokens, over the pipeline's per-replica prefill budget of "
                f"{token_budget} tokens",
                [
                    "--enable-chunked-prefill",
                    "--max-batch-input-tokens",
                    str(pipeline_config.runtime.max_batch_input_tokens),
                ],
                caller=caller,
                note=(
                    "To run this batch here, raise max_batch_input_tokens or "
                    "turn chunked prefill off."
                ),
            )
        if total_tokens_cap is None or not lengths:
            continue
        # The MLA prefill plan starts each row at a page-aligned offset and
        # clamps the last row's padded tail at the buffer end, so only its
        # raw tokens must fit. This equals serve's TotalContextTokenBudget
        # when the cap is a page multiple.
        resident_tokens = (
            sum(_align_up(length, page_size) for length in lengths[:-1])
            + lengths[-1]
        )
        if resident_tokens > total_tokens_cap:
            raise _unsupported(
                pipeline_config,
                "scheduling across batches",
                f"Replica {replica_idx}'s prompts occupy {resident_tokens} "
                f"tokens with each row but the last padded to a "
                f"{page_size}-token page, but the plan caps a replica's "
                f"resident context at {total_tokens_cap} tokens",
                ["--max-batch-total-tokens", str(total_tokens_cap)],
                caller=caller,
                note="To run this batch here, raise max_batch_total_tokens.",
            )


class GenerateIterator:
    """Steps a fixed batch of prompts through a pipeline, one execute at a time.

    GenerateIterator runs a pipeline's forward passes over exactly the batch
    you pass, with nothing added on top of the pipeline API, which also makes
    it useful for reproducing and localizing kernel and model accuracy bugs.
    Serving features (scheduling and in-flight batching, chunked prefill,
    prefix caching, LoRA, and structured output) are tested through
    ``max serve``. Requesting one that GenerateIterator cannot run raises an
    error that gives the matching ``max serve`` command.

    Constructing the batch claims KV cache for every row. Use it as a
    context manager so an exception releases the rows. To run contexts built
    elsewhere, such as by the pipeline's tokenizer, use :meth:`from_contexts`.

    A batch takes ``input_ids`` or ``messages``. With ``messages``, the
    pipeline's tokenizer builds each row as serve builds a chat request,
    applying the chat template and expanding each image's placeholders, so
    read each row's prompt length from :attr:`prompt_lengths`.

    In a batch built from ``input_ids`` or ``messages``, row ``i`` samples with
    ``seed + i * 2**32``. Serve deliberately makes requests with the same
    pinned seed draw identically. GenerateIterator instead gives each row its
    own stream, as Hugging Face users expect, so identical prompts draw
    different samples, a rerun reproduces them, and row 0 matches serve for
    the same seed. A prompt's draw still depends on its row index.
    :meth:`from_contexts` keeps each context's own seed, so it draws as
    serve does.

    On a :attr:`synchronous` pipeline, each :meth:`step` returns its own
    forward's tokens and the device is idle between steps. A pipeline that
    is not synchronous (an overlap pipeline with overlap active) returns
    the previous execute's tokens instead: the first step returns ``{}``, a
    row finishes one step after its last token is sampled, ``ignore_eos``
    takes ``max_new_tokens + 1`` steps, and a forward can still be running
    between steps and after :attr:`done`. Such
    a pipeline allows one open batch at a time, and if its execute raises,
    reload it: closing the batch frees the rows but cannot repair the
    pipeline's pending state.

    ``ignore_eos=True`` is the batch mode closest to Hugging Face
    ``generate``, which keeps finished rows in the batch as padding. Prefill shapes still
    differ, because MAX batches prompts ragged where Hugging Face pads them
    on the left, so compare logits with a tolerance.

    With prefix caching on, a later batch can reuse KV an earlier one
    committed and prefill fewer tokens. Turn prefix caching off in the KV
    cache config when prefill shapes must repeat across runs.

    On a pipeline that ran ``warmup_graph_capture()``, the first (prefill)
    step runs eagerly and later steps replay captured graphs, as long as
    each replica has at most ``min(max_batch_size, 128)`` rows. Warm up
    before creating a GenerateIterator. With device graph capture on but no
    warmup, GenerateIterator logs a warning and runs every step eagerly. A
    logits processor receives the captured output buffer, which the next
    replay overwrites, so it must copy what it keeps. ``output_logits``
    copies a float32 ``[vocab]`` logits row to host for every generated
    token, so host memory grows by ``4 * vocab`` bytes per token.

    .. Skipped: the example needs a loaded pipeline.
    .. skip: next

    .. code-block:: python

        with GenerateIterator(pipeline, input_ids, max_new_tokens=20) as batch:
            for step_tokens in batch:
                print(step_tokens)
        sequences = batch.sequences

    Args:
        pipeline: The pipeline to run.
        input_ids: One token-id prompt per row, as lists or a 2D array.
        attention_mask: Zeros mark padding in ``input_ids``, which
            GenerateIterator drops, since it runs prompts unpadded. A
            tokenizer returns it with the ids.
        messages: One conversation per row, instead of ``input_ids``. Each
            message is a
            :class:`~max.pipelines.modeling.types.TextGenerationRequestMessage`
            or a dict in the same shape, with an image part where each image
            goes.
        images: One list of images per conversation, matching its image
            parts in order, each as encoded bytes or a PIL image. A PIL
            image reaches the tokenizer as is, without a second decode.
            Needs an architecture that takes images.
        chat_template_kwargs: Keyword arguments passed to every row's chat
            template as given, such as ``{"enable_thinking": False}``. Unlike
            serve, ``reasoning_effort`` does not set the thinking switches.
            Needs ``messages``.
        max_new_tokens: The most tokens to generate for each row.
        min_new_tokens: The fewest tokens a row generates before EOS can end
            it. Defaults to the model's generation config.
        temperature: The sampling temperature. Defaults to the pipeline's
            runtime default, then the model's generation config.
        top_k: The top-k limit, with the same fallback as ``temperature``.
        top_p: The nucleus sampling threshold. Defaults to the model's
            generation config.
        min_p: The minimum token probability relative to the most likely
            token. Defaults to 0 (off), since the generation config does
            not set it.
        repetition_penalty: The divisor for logits of repeated tokens.
            Defaults to the model's generation config.
        frequency_penalty: The penalty proportional to a token's count.
            Defaults to 0 (off), since the generation config does not set
            it.
        presence_penalty: The flat penalty for a token already present.
            Defaults to 0 (off), since the generation config does not set
            it.
        ignore_eos: Keeps generating past EOS, so every row runs
            ``max_new_tokens`` tokens and the batch size never changes.
        stop_token_ids: Token ids that end a row, in addition to EOS.
        seed: The sampling seed for row 0. Row ``i`` uses
            ``seed + i * 2**32``.
        output_logits: Copies each row's next-token logits to host on every
            step, for :attr:`logits`. Needs a :attr:`synchronous` pipeline.
        output_logprobs: Records the log probability of each generated token
            and of this many top candidates, for :attr:`log_probs`. A model
            that does not compute log probabilities raises ``ValueError``.
        caller: What the user called, which errors name, such as
            ``"AutoPipeline"``.

    Raises:
        TypeError: If ``input_ids`` or ``attention_mask`` is not a list of
            integer lists, ``messages`` is not a list of conversations, or
            ``images`` is not a list of lists of encoded bytes or PIL
            images.
        ValueError: If the request needs a feature GenerateIterator leaves to
            ``max serve``: LoRA, a model that generates neither text nor
            images, a video model, a block diffusion model, video inputs,
            device graph capture with
            ``data_parallel_degree > 1`` on a warmed-up pipeline, or a batch
            that only fits by scheduling or chunked prefill (more rows than
            the planned max batch size, or a replica's prompts over its
            per-replica prefill budget or the batch total-token cap). An
            image generation model raises it pointing to
            :func:`~max.pipelines.lib.generate_images.generate_images`. Also
            raised for a prompt with no
            room to generate within the planned max length, an invalid
            sampling value, penalties on a pipeline built without
            ``enable_penalties``, ``max_new_tokens`` below 1 or below
            ``min_new_tokens``, an ``input_ids`` token id below 0 or past the
            model's vocabulary, ``attention_mask`` with a different shape
            than ``input_ids``, or ``ignore_eos`` with a row that does not fit
            or with speculative decoding. Also raised for both or neither of
            ``input_ids`` and ``messages``, ``images`` with another row count
            than ``messages``, images or video on an architecture that takes
            text only, a row with more images than the tokenizer's
            ``max_images_per_request``, a row whose image count differs from
            its image parts, an ``image_url`` part or an image part with its
            own image, ``images``,
            ``chat_template_kwargs`` or ``attention_mask`` without the
            matching prompt argument, or a ``chat_template_kwargs`` key the
            tokenizer sets itself. Also raised for ``output_logprobs``
            outside ``[0, 7]``,
            ``output_logprobs`` on a model that does not compute log
            probabilities, ``output_logits`` on a pipeline that is not
            synchronous, ``output_logits`` or ``output_logprobs`` with
            speculative decoding, or ``output_logprobs`` on an
            :class:`~max.pipelines.lib.OverlapTextGenerationPipeline`.
        RuntimeError: If the pipeline is not synchronous and another batch
            on it is still open.
    """

    _logits_recorders: list[_LogitsRecorder] | None
    _log_probs: list[LogProbabilities] | None

    def __init__(
        self,
        pipeline: BatchPipeline,
        input_ids: Iterable[Sequence[int]] | None = None,
        attention_mask: Iterable[Sequence[int]] | None = None,
        *,
        messages: Sequence[
            Sequence[TextGenerationRequestMessage | Mapping[str, object]]
        ]
        | None = None,
        images: Sequence[Sequence[bytes | Image.Image]] | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        repetition_penalty: float | None = None,
        ignore_eos: bool = False,
        stop_token_ids: Sequence[int] | None = None,
        seed: int = 0,
        output_logits: bool = False,
        output_logprobs: int = 0,
        caller: str = "generate()",
    ) -> None:
        pipeline_config = pipeline.pipeline_config
        context_type = _check_supported(pipeline_config, caller)
        rows: list[list[int]] | list[_Conversation]
        if messages is None:
            if input_ids is None:
                raise ValueError("Pass input_ids or messages.")
            if images is not None:
                raise ValueError(
                    "Pass images with messages, since each image needs an "
                    "image part in a message."
                )
            if chat_template_kwargs is not None:
                raise ValueError(
                    "Pass chat_template_kwargs with messages, since "
                    "input_ids get no chat template."
                )
            rows = _materialize_prompts(input_ids, attention_mask)
        else:
            if input_ids is not None:
                raise ValueError("Pass input_ids or messages, not both.")
            if attention_mask is not None:
                raise ValueError("Pass attention_mask only with input_ids.")
            if chat_template_kwargs is not None:
                taken = sorted(
                    _TOKENIZER_TEMPLATE_KEYS & chat_template_kwargs.keys()
                )
                if taken:
                    raise ValueError(
                        f"chat_template_kwargs cannot set {taken[0]!r}, "
                        "since the tokenizer sets it itself."
                    )
            max_images = getattr(
                pipeline.tokenizer, "max_images_per_request", None
            )
            rows = _materialize_conversations(
                pipeline_config,
                messages,
                images,
                caller,
                max_images if isinstance(max_images, int) else None,
            )
        penalties = {
            name: value
            for name, value, default in (
                ("repetition_penalty", repetition_penalty, 1.0),
                ("frequency_penalty", frequency_penalty, 0.0),
                ("presence_penalty", presence_penalty, 0.0),
            )
            if value is not None and value != default
        }
        if ignore_eos and pipeline_config.speculative is not None:
            raise ValueError(
                "ignore_eos cannot keep the batch size fixed with speculative "
                "decoding, because a step can add several tokens to a row, so "
                "rows reach max_new_tokens at different steps."
            )
        if max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be at least 1, was {max_new_tokens}."
            )
        sampling_params = _resolve_sampling_params(
            pipeline,
            SamplingParamsInput(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                ignore_eos=ignore_eos,
                stop_token_ids=(
                    None if stop_token_ids is None else list(stop_token_ids)
                ),
                seed=seed,
            ),
        )
        # After validation, so a bad penalty value reports itself.
        _check_penalties_enabled(pipeline_config, penalties)
        contexts = _build_contexts(
            pipeline,
            rows,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            tokenizer_builds_rows=messages is not None
            or issubclass(context_type, TextAndVisionContext),
            chat_template_kwargs=chat_template_kwargs,
        )
        # After the max length check: a too-long prompt also trips these
        # limits, and their advice cannot fix it.
        dp_degree = pipeline_config.model.data_parallel_degree
        _check_batch_fits(
            pipeline,
            [len(context.tokens) for context in contexts],
            dp_degree,
            caller,
        )
        _check_graph_capture(pipeline, dp_degree, caller)
        self._start(pipeline, contexts, dp_degree)
        _check_output_options(
            pipeline, self._synchronous, output_logits, output_logprobs, caller
        )
        if output_logits:
            self._logits_recorders = [_LogitsRecorder() for _ in contexts]
            for context, recorder in zip(
                contexts, self._logits_recorders, strict=True
            ):
                context.sampling_params.logits_processors = [
                    *(context.sampling_params.logits_processors or ()),
                    recorder,
                ]
        if output_logprobs:
            self._log_probs = [
                LogProbabilities(
                    token_log_probabilities=[],
                    top_log_probabilities=[],
                    sampled_token_ids=[],
                )
                for _ in contexts
            ]
            for context in contexts:
                context.log_probabilities = output_logprobs
        self._claim_rows()

    @classmethod
    def from_contexts(
        cls,
        pipeline: BatchPipeline,
        contexts: Iterable[TextContext],
        *,
        caller: str = "GenerateIterator.from_contexts()",
    ) -> GenerateIterator:
        """Claims a batch over contexts already built, one row per context.

        Each context keeps its own sampling params, seed, logits processors,
        stop criteria and max length, so this runs the contexts a pipeline's
        tokenizer builds, image contexts included. On a speculative
        pipeline one step can return several tokens per row. It runs
        base-model contexts and contexts that name an active LoRA adapter. The batch records no :attr:`logits` or
        :attr:`log_probs`.

        Args:
            pipeline: The pipeline to run.
            contexts: One built context per row, for example from
                ``tokenizer.new_context``, claimed in order.
            caller: What the user called, which errors name, such as
                ``"max generate"``.

        Returns:
            A batch with every row claimed.

        Raises:
            ValueError: If a context names a LoRA adapter that is loaded but
                not active, or for device graph capture with
                ``data_parallel_degree > 1`` on a warmed-up pipeline, or if
                the batch does not fit the pipeline's plan, as described for
                :class:`GenerateIterator`.
            RuntimeError: If the pipeline is not synchronous and another
                batch on it is still open.
        """
        contexts = list(contexts)
        _check_adapters_active(pipeline, contexts, caller)
        dp_degree = pipeline.pipeline_config.model.data_parallel_degree
        _check_batch_fits(
            pipeline,
            [len(context.tokens) for context in contexts],
            dp_degree,
            caller,
        )
        _check_graph_capture(pipeline, dp_degree, caller)
        batch = cls.__new__(cls)
        batch._start(pipeline, contexts, dp_degree)
        batch._claim_rows()
        return batch

    def _start(
        self,
        pipeline: BatchPipeline,
        contexts: list[TextContext],
        dp_degree: int,
    ) -> None:
        synchronous = not (
            isinstance(pipeline, _OverlapSource) and pipeline.overlap_active
        )
        if not synchronous:
            open_batch_ref = _open_iterators.get(pipeline)
            open_batch = (
                open_batch_ref() if open_batch_ref is not None else None
            )
            if open_batch is not None and not open_batch.done:
                raise RuntimeError(
                    "The pipeline is already running another batch, and a "
                    "pipeline that is not synchronous returns each step's "
                    "results to whichever batch steps next. Finish or close "
                    "that batch first."
                )
        self._pipeline: BatchPipeline | None = pipeline
        self._dp_degree = dp_degree
        self._synchronous = synchronous
        self._contexts = contexts
        self._prompts = [
            [int(token) for token in context.tokens.prompt]
            for context in contexts
        ]
        self._generated: list[list[int]] = [[] for _ in contexts]
        self._logits_recorders = None
        self._log_probs = None
        # Live rows in input order, mapped to their DP replica. Replicas
        # need no padding: GenerateIterator rejects data parallelism on a
        # warmed-up pipeline, and eager steps accept unequal replicas.
        self._live: dict[int, int] = {}

    def _claim_rows(self) -> None:
        pipeline = self._pipeline
        assert pipeline is not None
        try:
            for row, context in enumerate(self._contexts):
                replica_idx = row % self._dp_degree
                pipeline.kv_manager.claim(context, replica_idx=replica_idx)
                self._live[row] = replica_idx
        except BaseException:
            self.close()
            raise
        if not self._synchronous:
            _open_iterators[pipeline] = weakref.ref(self)

    def __enter__(self) -> GenerateIterator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[dict[int, list[int]]]:
        """Steps until every row finishes, yielding each :meth:`step`'s tokens."""
        while not self.done:
            yield self.step()

    @property
    def synchronous(self) -> bool:
        """Whether each step returns its own forward's tokens, with no lag."""
        return self._synchronous

    @property
    def done(self) -> bool:
        """Whether every row has finished or been removed."""
        return not self._live

    @property
    def sequences(self) -> list[list[int]]:
        """Each row's prompt followed by its new tokens so far, in input order."""
        return [
            [*prompt, *generated]
            for prompt, generated in zip(
                self._prompts, self._generated, strict=True
            )
        ]

    @property
    def prompt_lengths(self) -> list[int]:
        """Each row's prompt length, in input order."""
        return [len(prompt) for prompt in self._prompts]

    @property
    def logits(self) -> list[npt.NDArray[np.float32]]:
        """The next-token logits behind each row's new tokens, in input order.

        Row ``i`` has one ``[vocab]`` row per new token, and ``logits[i][t]``
        holds the raw model output that produced the row's ``t``-th new
        token, before temperature, top-k, penalties or token masks. A row
        with no new tokens has shape ``[0, 0]``.

        Raises:
            RuntimeError: If the batch was built without ``output_logits``,
                as every :meth:`from_contexts` batch is.
        """
        if self._logits_recorders is None:
            raise RuntimeError(
                "Build the batch with output_logits=True. A batch from "
                "GenerateIterator.from_contexts records no logits, so read "
                "them through a logits processor on each context."
            )
        return [
            np.stack(recorder.logits[: len(tokens)])
            if tokens
            else np.empty((0, 0), dtype=np.float32)
            for recorder, tokens in zip(
                self._logits_recorders, self._generated, strict=True
            )
        ]

    @property
    def log_probs(self) -> list[LogProbabilities]:
        """The log probabilities behind each row's new tokens, in input order.

        For row ``i``, ``token_log_probabilities[t]`` is the log probability
        of the row's ``t``-th new token, and ``top_log_probabilities[t]``
        maps the ``output_logprobs`` most likely token ids, plus that token
        if it is not among them, to their log probabilities.
        ``sampled_token_ids`` holds the row's new tokens. They match
        ``log_softmax(logits[i][t])`` when the sampler applies no penalties
        and no ``min_new_tokens`` mask. With the default sampler on the
        model device, both edit the logits before log probabilities are
        taken. With ``enable_echo`` or ``sample_on_host``, log probabilities
        come from the unpenalized logits.

        Raises:
            RuntimeError: If the batch was built without ``output_logprobs``,
                as every :meth:`from_contexts` batch is.
        """
        if self._log_probs is None:
            raise RuntimeError(
                "Build the batch with output_logprobs > 0. A batch from "
                "GenerateIterator.from_contexts records no log probabilities."
            )
        return [
            LogProbabilities(
                token_log_probabilities=log_probs.token_log_probabilities[
                    : len(tokens)
                ],
                top_log_probabilities=log_probs.top_log_probabilities[
                    : len(tokens)
                ],
                sampled_token_ids=log_probs.sampled_token_ids[: len(tokens)],
            )
            for log_probs, tokens in zip(
                self._log_probs, self._generated, strict=True
            )
        ]

    def step(self) -> dict[int, list[int]]:
        """Runs exactly one ``pipeline.execute`` over the live rows.

        Rows the execute reports done are released before this returns. On a
        pipeline that is not synchronous the tokens lag one step, as
        :class:`GenerateIterator` describes.

        Returns:
            The tokens the execute reported for each live row, keyed by row
            index.

        Raises:
            RuntimeError: If no rows are live, or if the pipeline returns log
                probabilities or logits rows that do not line up one per
                generated token.
        """
        if self.done:
            raise RuntimeError("GenerateIterator has no live rows to step.")
        pipeline = self._pipeline
        assert pipeline is not None
        # Fresh lists every step: OverlapTextGenerationPipeline keeps a
        # reference to the previous step's batch, which must not change
        # underneath it.
        batches: list[list[TextContext]] = [[] for _ in range(self._dp_degree)]
        for row, replica_idx in self._live.items():
            context = self._contexts[row]
            # A KV connector's onload lands off the forward stream, so the
            # scheduler holds the row back until the onload lands.
            # GenerateIterator waits instead.
            pipeline.kv_manager.alloc(context).synchronize()
            batches[replica_idx].append(context)
        # Rebuilt each step so decode runs as TG, as under the scheduler.
        # Reusing one inputs object would leave every step CE.
        outputs = pipeline.execute(TextGenerationInputs(batches=batches))

        # Looking up live rows only drops the extra output a pipeline that
        # is not synchronous emits for a row already released.
        step_tokens: dict[int, list[int]] = {}
        for row in list(self._live):
            output = outputs.get(self._contexts[row].request_id)
            if output is None:
                continue
            tokens = [int(token) for token in output.tokens]
            self._generated[row].extend(tokens)
            self._record_log_probs(row, output, tokens)
            self._check_logits_aligned(row)
            step_tokens[row] = tokens
            if output.is_done:
                self._release(row)
        return step_tokens

    def _record_log_probs(
        self, row: int, output: TextGenerationOutput, tokens: list[int]
    ) -> None:
        if self._log_probs is None:
            return
        entries = output.log_probabilities or []
        if len(entries) != len(tokens):
            raise RuntimeError(
                f"Row {row} got log probabilities for {len(entries)} of "
                f"{len(tokens)} tokens. The model may not compute them."
            )
        sampled = [
            token for entry in entries for token in entry.sampled_token_ids
        ]
        if sampled != tokens:
            raise RuntimeError(
                f"Row {row} got log probabilities for {sampled} but "
                f"generated {tokens}."
            )
        row_log_probs = self._log_probs[row]
        for entry in entries:
            row_log_probs.token_log_probabilities.extend(
                entry.token_log_probabilities
            )
            row_log_probs.top_log_probabilities.extend(
                entry.top_log_probabilities
            )
            row_log_probs.sampled_token_ids.extend(entry.sampled_token_ids)

    def _check_logits_aligned(self, row: int) -> None:
        if self._logits_recorders is None:
            return
        num_logits = len(self._logits_recorders[row].logits)
        num_tokens = len(self._generated[row])
        if num_logits != num_tokens:
            raise RuntimeError(
                f"Row {row} has {num_logits} logits rows for {num_tokens} "
                "tokens. The pipeline did not run the logits processors once "
                "per sampled token."
            )

    def remove(self, row: int) -> None:
        """Releases ``row`` so later steps run without it.

        Removing a row that already finished is a no-op.

        Args:
            row: The input index of the row to remove.

        Raises:
            IndexError: If ``row`` is not an input index.
        """
        if not 0 <= row < len(self._contexts):
            raise IndexError(
                f"Row {row} is out of range for {len(self._contexts)} rows."
            )
        if row in self._live:
            self._release(row)

    def close(self) -> None:
        """Releases every live row, then drops the pipeline reference.

        Safe to call more than once. Every row is attempted even if one
        fails, and the first error is re-raised afterwards with the
        pipeline still held.
        """
        first_error: BaseException | None = None
        for row in list(self._live):
            try:
                self._release(row)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        # A caller that keeps the batch for its sequences must not keep the
        # pipeline's device memory alive.
        self._pipeline = None

    def _release(self, row: int) -> None:
        context = self._contexts[row]
        pipeline = self._pipeline
        assert pipeline is not None
        # Not a double free: the pipeline's release leaves the paged KV
        # cache alone and frees what the manager cannot, such as recurrent
        # state slots and encoder-cache refs.
        if pipeline.kv_manager.contains(context):
            pipeline.kv_manager.release(context)
        pipeline.release(context.request_id)
        # Dropped last, so a release that raises leaves the row to retry.
        del self._live[row]


def generate(
    pipeline: BatchPipeline,
    input_ids: Iterable[Sequence[int]] | None = None,
    attention_mask: Iterable[Sequence[int]] | None = None,
    *,
    messages: Sequence[
        Sequence[TextGenerationRequestMessage | Mapping[str, object]]
    ]
    | None = None,
    images: Sequence[Sequence[bytes | Image.Image]] | None = None,
    chat_template_kwargs: Mapping[str, object] | None = None,
    max_new_tokens: int,
    min_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    ignore_eos: bool = False,
    stop_token_ids: Sequence[int] | None = None,
    seed: int = 0,
    output_logits: bool = False,
    output_logprobs: int = 0,
    caller: str = "generate()",
) -> GenerateResult:
    """Generates tokens for a fixed batch of prompts, run as one batch.

    Steps a :class:`GenerateIterator` until every row finishes. The arguments
    and their defaults are the same as for :class:`GenerateIterator`, and the
    sampling arguments match the ``max generate`` flags of the same name.

    Args:
        pipeline: The pipeline to run.
        input_ids: One token-id prompt per row, as lists or a 2D array.
        attention_mask: Zeros mark padding in ``input_ids`` to drop.
        messages: One conversation per row, instead of ``input_ids``.
        images: One list of images per conversation, matching its image
            parts in order, each as encoded bytes or a PIL image.
        chat_template_kwargs: Keyword arguments for every row's chat
            template. Needs ``messages``.
        max_new_tokens: The most tokens to generate for each row.
        min_new_tokens: The fewest tokens a row generates before EOS can end
            it.
        temperature: The sampling temperature.
        top_k: The top-k limit.
        top_p: The nucleus sampling threshold.
        min_p: The minimum token probability relative to the most likely
            token.
        frequency_penalty: The penalty proportional to a token's count.
        presence_penalty: The flat penalty for a token already present.
        repetition_penalty: The divisor for logits of repeated tokens.
        ignore_eos: Keeps generating past EOS, so the batch size never
            changes.
        stop_token_ids: Token ids that end a row, in addition to EOS.
        seed: The sampling seed for row 0. Row ``i`` uses
            ``seed + i * 2**32``.
        output_logits: Returns each row's next-token logits, as
            :attr:`GenerateIterator.logits`.
        output_logprobs: Returns the log probabilities of each token and of
            this many top candidates, as :attr:`GenerateIterator.log_probs`.
        caller: What the user called, which errors name.

    Returns:
        Each row's prompt followed by its new tokens, in input order, with
        each prompt's length and the requested logits and log probabilities.

    Raises:
        TypeError: If ``input_ids``, ``attention_mask``, ``messages`` or
            ``images`` has the wrong shape, as described for
            :class:`GenerateIterator`.
        ValueError: If the request needs a feature GenerateIterator leaves to
            ``max serve``, such as device graph capture with
            ``data_parallel_degree > 1`` on a warmed-up pipeline, the batch
            does not fit the pipeline's plan, or the pipeline cannot return
            the requested outputs, as described for :class:`GenerateIterator`.
        RuntimeError: If the pipeline is not synchronous and another batch
            on it is still open, or if the pipeline returns log
            probabilities or logits rows that do not line up one per
            generated token.
    """
    with GenerateIterator(
        pipeline,
        input_ids,
        attention_mask,
        messages=messages,
        images=images,
        chat_template_kwargs=chat_template_kwargs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
        ignore_eos=ignore_eos,
        stop_token_ids=stop_token_ids,
        seed=seed,
        output_logits=output_logits,
        output_logprobs=output_logprobs,
        caller=caller,
    ) as batch:
        while not batch.done:
            batch.step()
        return GenerateResult(
            sequences=batch.sequences,
            prompt_lengths=batch.prompt_lengths,
            logits=batch.logits if output_logits else None,
            log_probs=batch.log_probs if output_logprobs else None,
        )
