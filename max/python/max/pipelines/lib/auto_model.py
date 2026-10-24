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

"""Loads a text or image generation model and its tokenizer in-process.

The defaults turn off serving features that change how a text run executes:
the overlap scheduler returns each step's tokens one step late, device graph
capture replays recorded graphs, prefix caching and the vision encoder cache
make a run depend on earlier ones, and chunked prefill splits a prompt the
caller passed whole.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from types import TracebackType
from typing import overload

from max.driver import accelerator_api
from max.pipelines.context import PixelContext, TextContext
from max.pipelines.modeling.types import (
    PipelineTask,
    TextGenerationRequestMessage,
)
from PIL import Image
from transformers import AutoTokenizer as _HFAutoTokenizer
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .config import PipelineArgs, PipelineConfig
from .generate import (
    GenerateIterator,
    GenerateResult,
    _check_supported,
    generate,
)
from .generate_images import (
    GenerateImagesResult,
    _check_image_model,
    _is_image_generation_model,
    generate_images,
)
from .pipeline_variants import PixelGenerationPipeline
from .pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from .pipeline_variants.text_generation import TextGenerationPipelineInterface
from .pixel_tokenizer import PixelGenerationTokenizer
from .registry import PIPELINE_REGISTRY

__all__ = ["AutoModel", "AutoTokenizer"]

_logger = logging.getLogger("max.pipelines")


_DEBUG_DEFAULTS: dict[str, object] = {
    "enable_overlap_scheduler": False,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "vision_cache_utilization": 0.0,
}

# ``PipelineArgs.from_flat_kwargs`` lets the flat debug defaults win over the
# values these keys carry, silently dropping the caller's settings.
_CLOBBERED_SECTIONS = frozenset({"runtime", "kv_cache", "model", "config_file"})


def _clobbered_sections(config_overrides: Mapping[str, object]) -> list[str]:
    sections = set(_CLOBBERED_SECTIONS & config_overrides.keys())
    model_override = config_overrides.get("model_override")
    if isinstance(model_override, list):
        for entry in model_override:
            # An entry reads ``component.field=value``.
            field = str(entry).partition("=")[0].partition(".")[2]
            if field in _CLOBBERED_SECTIONS:
                sections.add(field)
    return sorted(sections)


def _build_config(
    model_path: str,
    max_batch_size: int,
    device_graph_capture: bool,
    config_overrides: Mapping[str, object],
) -> tuple[PipelineConfig, bool]:
    """Returns the resolved config and whether the model generates images."""
    flat: dict[str, object] = {
        **_DEBUG_DEFAULTS,
        "device_graph_capture": device_graph_capture,
        "max_batch_size": max_batch_size,
    }
    if (
        device_graph_capture
        and config_overrides.get("enable_overlap_scheduler") is not True
    ):
        # Capture turns the overlap scheduler on, so this keeps each step
        # synchronous unless the caller asks for overlap.
        flat["disable_overlap"] = True
    flat.update(config_overrides)
    pipeline_args = PipelineArgs.from_flat_kwargs(model_path=model_path, **flat)
    if _is_image_generation_model(pipeline_args):
        pipeline_config = PipelineConfig.from_args(pipeline_args)
        _check_image_model(pipeline_config, "AutoModel")
        return pipeline_config, True
    if (
        accelerator_api() == "hip"
        and "disable_vendor_blas_fallback" not in config_overrides
    ):
        # Serve turns the vendor BLAS fallback off whenever it turns capture
        # on, so this compiles the AMD matmul kernels serve would. Serve
        # never captures an image model, so those keep the fallback.
        pipeline_args = pipeline_args.model_copy(
            update={
                "runtime": pipeline_args.runtime.model_copy(
                    update={"disable_vendor_blas_fallback": True}
                )
            }
        )
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    _check_supported(pipeline_config, "AutoModel")
    # The registry falls back to an architecture of another task, such as
    # embeddings, and fails partway through building it.
    arch = PIPELINE_REGISTRY.architecture_for_config(
        pipeline_config, task=PipelineTask.TEXT_GENERATION
    )
    if arch is not None and arch.task != PipelineTask.TEXT_GENERATION:
        raise ValueError(
            "AutoModel runs text and image generation models, and "
            f"{model_path} loads {arch.name} for "
            f"{arch.task.value.replace('_', ' ')}."
        )
    return pipeline_config, False


@dataclasses.dataclass(frozen=True)
class _ResolvedModel:
    """What :class:`AutoModel` loads, resolved before any weights load."""

    model_path: str
    pipeline_config: PipelineConfig
    generates_images: bool
    device_graph_capture: bool


def _resolve(
    model_path: str,
    max_batch_size: int,
    devices: str | None,
    device_graph_capture: bool,
    prefer_module_v3: bool,
    config_overrides: Mapping[str, object],
) -> _ResolvedModel:
    """Checks :class:`AutoModel`'s arguments and resolves its config."""
    config_overrides = dict(config_overrides)
    if devices is not None:
        if "device_specs" in config_overrides:
            raise TypeError("Pass devices or device_specs, not both.")
        # The config parses the same strings as the CLI's --devices.
        config_overrides["device_specs"] = devices
    config_overrides["prefer_module_v3"] = prefer_module_v3
    model_override = config_overrides.get("model_override")
    if isinstance(model_override, Iterable) and not isinstance(
        model_override, str
    ):
        # PipelineArgs accepts any iterable here. As a list, a tuple
        # cannot skip the check below and a generator survives it.
        config_overrides["model_override"] = list(model_override)
    if sections := _clobbered_sections(config_overrides):
        raise TypeError(
            f"Pass the fields of {sections} as flat keyword arguments, "
            "since AutoModel cannot merge a whole section with its debug "
            "defaults."
        )
    # A pipeline from an earlier AutoModel can outlive close() in a
    # reference cycle, such as the cycle a caught exception creates.
    # Collect it before memory planning reads free device memory.
    gc.collect()
    pipeline_config, generates_images = _build_config(
        model_path, max_batch_size, device_graph_capture, config_overrides
    )
    return _ResolvedModel(
        model_path, pipeline_config, generates_images, device_graph_capture
    )


class AutoTokenizer:
    """Loads the Hugging Face tokenizer serve uses to encode a model's text.

    Calling it returns the ``input_ids`` and ``attention_mask`` that
    :meth:`AutoModel.generate` takes, with special tokens added as serve adds
    them to a completion prompt.

    .. Skipped: the example downloads a model.
    .. skip: next

    .. code-block:: python

        tokenizer = AutoTokenizer("HuggingFaceTB/SmolLM2-135M")
        model = AutoModel("HuggingFaceTB/SmolLM2-135M", max_batch_size=2)
        inputs = tokenizer(["The capital of France is", "2 + 2 ="])
        outputs = model.generate(**inputs, max_new_tokens=16)
        print(tokenizer.batch_decode(outputs.sequences))

    Args:
        model_path: A Hugging Face repo ID or a local model directory.
        revision: The repo revision to load.
        trust_remote_code: Whether to run tokenizer code from the repo.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self._delegate: PreTrainedTokenizerBase = (
            _HFAutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        )

    def __call__(self, text: str | Sequence[str]) -> BatchEncoding:
        """Returns ``input_ids`` and ``attention_mask`` for ``text``.

        Pass a list of prompts for a batch. The prompts stay unpadded.
        """
        return self._delegate(
            text if isinstance(text, str) else list(text),
            add_special_tokens=True,
            return_token_type_ids=False,
        )

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        """Returns the text for ``token_ids``."""
        text = self._delegate.decode(
            list(token_ids), skip_special_tokens=skip_special_tokens
        )
        assert isinstance(text, str)
        return text

    def batch_decode(
        self,
        sequences: Iterable[Sequence[int]],
        *,
        skip_special_tokens: bool = False,
    ) -> list[str]:
        """Returns the text for each token-id list in ``sequences``."""
        return [
            self.decode(token_ids, skip_special_tokens=skip_special_tokens)
            for token_ids in sequences
        ]


class AutoModel:
    """Loads a text or image generation model in-process.

    Unlike Hugging Face's base-model ``AutoModel``, this one generates. The
    Hugging Face equivalent is ``AutoModelForCausalLM``. Serving features are
    tested through ``max serve``.

    An image generation model, such as FLUX.2, runs through
    :meth:`generate` with ``prompt``. Its pipeline takes one prompt per
    call, so ``max_batch_size`` stays 1 and device graph capture does not
    apply.

    .. Skipped: the example downloads a model.
    .. skip: next

    .. code-block:: python

        model = AutoModel("black-forest-labs/FLUX.2-klein-4B")
        result = model.generate(
            prompt=["a red fox in snow", "a lighthouse at dusk"],
            height=512,
            width=512,
            seed=0,
        )
        result.images[0][0].save("fox.png")

    The config turns off the overlap scheduler, device graph capture, prefix
    caching, the vision encoder cache and chunked prefill. On AMD it also
    turns off the vendor BLAS fallback for a text model, as serve does with
    capture on, so matmuls compile to serve's kernels. As in serve, the
    architecture's ``required_arguments`` win over the overrides.

    References to the pipeline, its KV cache manager, an open
    :class:`~max.pipelines.lib.generate.GenerateIterator`, or a live traceback
    from inside the pipeline keep device memory alive past :meth:`close`.

    Args:
        model_path: A Hugging Face repo ID or a local model directory.
        max_batch_size: The most prompts per data-parallel replica, fixed at
            construction. With ``device_graph_capture=True``, warmup captures
            decode graphs for every batch size up to this value, so a larger
            value costs more warmup time and device memory.
        devices: The devices to load on, written as the CLI's ``--devices``
            takes them, such as ``"cpu"``, ``"gpu:0"`` or ``"gpu:0,1"``.
            Defaults to the first visible device.
        device_graph_capture: Builds the overlap pipeline and runs its
            ``warmup_graph_capture()``, so decode steps replay captured
            graphs. Steps stay synchronous unless
            ``enable_overlap_scheduler=True`` is also passed.
        prefer_module_v3: Loads the architecture's ModuleV3 version, built
            with ``max.experimental.nn``, where it has one.
        **config_overrides: Flat :class:`~max.pipelines.lib.PipelineArgs`
            field names, such as ``device_specs`` or
            ``enable_prefix_caching``. They win over the debug defaults.

    Raises:
        TypeError: If both ``devices`` and ``device_specs`` are passed, if an
            override passes a ``runtime``, ``kv_cache`` or
            ``model`` section, or a ``config_file``, instead of flat fields,
            or a ``model_override`` entry sets one of them whole.
        ValueError: If the config needs a feature that only ``max serve``
            runs, such as LoRA, a model that generates neither text nor
            images, a video model or a block diffusion model, if
            ``device_graph_capture=True`` builds a pipeline that
            cannot capture graphs, or if an image generation model gets
            ``device_graph_capture=True`` or a ``max_batch_size`` other than
            1.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_batch_size: int = 1,
        devices: str | None = None,
        device_graph_capture: bool = False,
        prefer_module_v3: bool = False,
        **config_overrides: object,
    ) -> None:
        self._load(
            _resolve(
                model_path,
                max_batch_size,
                devices,
                device_graph_capture,
                prefer_module_v3,
                config_overrides,
            )
        )

    def _load(self, resolved: _ResolvedModel) -> None:
        model_path = resolved.model_path
        self._pipeline_config = resolved.pipeline_config
        self._pipeline: TextGenerationPipelineInterface[TextContext] | None = (
            None
        )
        self._image_pipeline: PixelGenerationPipeline[PixelContext] | None = (
            None
        )
        self._image_tokenizer: PixelGenerationTokenizer | None = None
        if resolved.generates_images:
            if resolved.device_graph_capture:
                raise ValueError(
                    "device_graph_capture=True does not apply to "
                    f"{model_path}, an image generation model, since its "
                    "pipeline runs the whole denoising loop in one call."
                )
            max_batch_size = self._pipeline_config.runtime.max_batch_size
            if max_batch_size != 1:
                raise ValueError(
                    f"max_batch_size must be 1 for {model_path}, an image "
                    "generation model, since its pipeline runs one prompt "
                    f"per call. Got {max_batch_size}."
                )
            tokenizer, image_pipeline = PIPELINE_REGISTRY.retrieve(
                self._pipeline_config, task=PipelineTask.PIXEL_GENERATION
            )
            assert isinstance(image_pipeline, PixelGenerationPipeline)
            assert isinstance(tokenizer, PixelGenerationTokenizer)
            self._image_pipeline = image_pipeline
            self._image_tokenizer = tokenizer
            return
        _, pipeline = PIPELINE_REGISTRY.retrieve(self._pipeline_config)
        if not isinstance(pipeline, TextGenerationPipelineInterface):
            pipeline_class = type(pipeline).__name__
            # The traceback's frame would hold the pipeline's device memory.
            del pipeline
            gc.collect()
            raise ValueError(
                "AutoModel runs text and image generation models, and "
                f"{model_path} loads as {pipeline_class}."
            )
        self._pipeline = pipeline
        if not resolved.device_graph_capture:
            return
        try:
            if not isinstance(pipeline, OverlapTextGenerationPipeline):
                raise ValueError(
                    "device_graph_capture=True needs an "
                    f"OverlapTextGenerationPipeline, but {model_path} builds "
                    f"{type(pipeline).__name__}, which cannot capture graphs."
                )
            start = time.monotonic()
            pipeline.warmup_graph_capture()
            _logger.info(
                "Device graph capture warmup completed in %.2fs "
                "(max_batch_size=%d).",
                time.monotonic() - start,
                pipeline.max_batch_size,
            )
        except BaseException:
            # The traceback's warmup frames hold the pipeline past close().
            del pipeline
            self.close()
            raise

    @property
    def pipeline(self) -> TextGenerationPipelineInterface[TextContext]:
        """The loaded text pipeline.

        Raises:
            ValueError: If the model generates images.
            RuntimeError: If the model has been closed.
        """
        if self._image_pipeline is not None:
            raise ValueError(
                f"AutoModel loaded {self._pipeline_config.models.model_name}, "
                "an image generation model. Use AutoModel.generate(prompt=...) "
                "for it."
            )
        if self._pipeline is None:
            raise RuntimeError("AutoModel is closed. Create a new one.")
        return self._pipeline

    @property
    def pipeline_config(self) -> PipelineConfig:
        """The resolved config that built the pipeline."""
        return self._pipeline_config

    @overload
    def generate(
        self,
        *,
        prompt: str | Sequence[str],
        negative_prompt: str | Sequence[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        true_cfg_scale: float | None = None,
        num_images_per_prompt: int = 1,
        seed: int = 0,
    ) -> GenerateImagesResult: ...

    @overload
    def generate(
        self,
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
    ) -> GenerateResult: ...

    def generate(
        self,
        input_ids: Iterable[Sequence[int]] | None = None,
        attention_mask: Iterable[Sequence[int]] | None = None,
        *,
        prompt: str | Sequence[str] | None = None,
        messages: Sequence[
            Sequence[TextGenerationRequestMessage | Mapping[str, object]]
        ]
        | None = None,
        images: Sequence[Sequence[bytes | Image.Image]] | None = None,
        chat_template_kwargs: Mapping[str, object] | None = None,
        max_new_tokens: int | None = None,
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
        negative_prompt: str | Sequence[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        true_cfg_scale: float | None = None,
        num_images_per_prompt: int = 1,
    ) -> GenerateResult | GenerateImagesResult:
        """Generates text or images, whichever the loaded model makes.

        For a text model, it runs :func:`~max.pipelines.lib.generate.generate`
        on :attr:`pipeline` with the same arguments, so
        ``model.generate(**tokenizer(prompts), max_new_tokens=16)`` works with
        an :class:`AutoTokenizer`. Pass ``messages`` and ``images`` instead of
        ``input_ids`` for chat or image prompts, and read each prompt's length
        from the result's ``prompt_lengths``. ``chat_template_kwargs`` applies
        to every ``messages`` row, such as ``{"enable_thinking": False}``.

        For an image model, pass ``prompt`` and the image arguments, and it
        runs :func:`~max.pipelines.lib.generate_images.generate_images` with
        them. An argument counts as passed when it differs from its default.

        Raises:
            TypeError: If a text model gets no ``max_new_tokens`` or an image
                model no ``prompt``, or as described for the function it
                runs.
            ValueError: If an argument is for the other kind of model, or as
                described for the function it runs.
            RuntimeError: If the model has been closed, or as described for
                the function it runs.
        """
        if self._image_pipeline is None and self._pipeline is None:
            raise RuntimeError("AutoModel is closed. Create a new one.")
        text_only: dict[str, tuple[object, object]] = {
            "input_ids": (input_ids, None),
            "messages": (messages, None),
            "attention_mask": (attention_mask, None),
            "images": (images, None),
            "chat_template_kwargs": (chat_template_kwargs, None),
            "max_new_tokens": (max_new_tokens, None),
            "min_new_tokens": (min_new_tokens, None),
            "temperature": (temperature, None),
            "top_k": (top_k, None),
            "top_p": (top_p, None),
            "min_p": (min_p, None),
            "frequency_penalty": (frequency_penalty, None),
            "presence_penalty": (presence_penalty, None),
            "repetition_penalty": (repetition_penalty, None),
            "ignore_eos": (ignore_eos, False),
            "stop_token_ids": (stop_token_ids, None),
            "output_logits": (output_logits, False),
            "output_logprobs": (output_logprobs, 0),
        }
        image_only: dict[str, tuple[object, object]] = {
            "prompt": (prompt, None),
            "negative_prompt": (negative_prompt, None),
            "height": (height, None),
            "width": (width, None),
            "num_inference_steps": (num_inference_steps, None),
            "guidance_scale": (guidance_scale, None),
            "true_cfg_scale": (true_cfg_scale, None),
            "num_images_per_prompt": (num_images_per_prompt, 1),
        }
        if self._image_pipeline is not None:
            assert self._image_tokenizer is not None
            self._reject_arguments(
                text_only, "text", "Pass the text as prompt= instead."
            )
            if prompt is None:
                raise TypeError(
                    "AutoModel.generate() on an image generation model needs "
                    "prompt=."
                )
            return generate_images(
                self._image_pipeline,
                self._image_tokenizer,
                prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                num_images_per_prompt=num_images_per_prompt,
                seed=seed,
                caller="AutoModel",
            )
        self._reject_arguments(
            image_only, "image", "Pass input_ids or messages instead."
        )
        if max_new_tokens is None:
            raise TypeError(
                "AutoModel.generate() missing 1 required keyword-only "
                "argument: 'max_new_tokens'"
            )
        return generate(
            self.pipeline,
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
            caller="AutoModel",
        )

    def _reject_arguments(
        self,
        arguments: Mapping[str, tuple[object, object]],
        kind: str,
        hint: str,
    ) -> None:
        """Raises if the caller passed any of ``arguments``.

        Each maps to its value and its default. They are for ``kind``
        generation models, and the loaded model is the other kind. ``hint``
        follows when the rejected arguments carry the prompts.
        """
        given = [
            name
            for name, (value, default) in arguments.items()
            # A None default compares by identity, since an array argument
            # has no single truth value.
            if (value is not None if default is None else value != default)
        ]
        if not given:
            return
        loaded = "an image" if kind == "text" else "a text"
        verb = "is" if len(given) == 1 else "are"
        message = (
            f"{', '.join(given)} {verb} for {kind} generation models, and "
            f"AutoModel loaded {self._pipeline_config.models.model_name}, "
            f"{loaded} generation model."
        )
        if {"input_ids", "messages", "prompt"} & set(given):
            message += f" {hint}"
        raise ValueError(message)

    def generate_iter(
        self,
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
    ) -> GenerateIterator:
        """Opens a :class:`~max.pipelines.lib.generate.GenerateIterator` on :attr:`pipeline`.

        Takes the same keyword arguments. Use it as a context manager so its
        rows are released.

        Raises:
            TypeError: As described for
                :class:`~max.pipelines.lib.generate.GenerateIterator`.
            ValueError: If the model generates images, or as described for
                :class:`~max.pipelines.lib.generate.GenerateIterator`.
            RuntimeError: If the model has been closed, or as described for
                :class:`~max.pipelines.lib.generate.GenerateIterator`.
        """
        return GenerateIterator(
            self.pipeline,
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
            caller="AutoModel",
        )

    def close(self) -> None:
        """Releases the pipeline, after shutting down a text model's KV cache.

        Calling it again does nothing. The pipeline is released even if the
        shutdown raises.
        """
        if self._image_pipeline is not None:
            # An image pipeline holds no KV cache.
            self._image_pipeline = None
            self._image_tokenizer = None
            gc.collect()
            return
        if self._pipeline is None:
            return
        pipeline, self._pipeline = self._pipeline, None
        try:
            pipeline.kv_manager.shutdown()
        finally:
            del pipeline
            # Free the device buffers now, not at the next cyclic collection.
            gc.collect()

    def __enter__(self) -> AutoModel:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
