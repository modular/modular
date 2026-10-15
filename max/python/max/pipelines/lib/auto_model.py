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

"""Loads a text generation model and its tokenizer in-process.

The defaults turn off serving features that change how a run executes: the
overlap scheduler returns each step's tokens one step late, device graph
capture replays recorded graphs, prefix caching makes a run depend on earlier
ones, and chunked prefill splits a prompt the caller passed whole.
"""

from __future__ import annotations

import gc
import logging
import time
from collections.abc import Iterable, Mapping, Sequence
from types import TracebackType

from max.driver import accelerator_api
from max.pipelines.context import TextContext
from max.pipelines.modeling.types import PipelineTask
from pydantic import BaseModel
from transformers import AutoTokenizer as _HFAutoTokenizer
from transformers import BatchEncoding, PreTrainedTokenizerBase

from .config import PipelineArgs, PipelineConfig
from .generate import Batch, GenerateResult, _check_supported, generate
from .pipeline_variants.overlap_text_generation import (
    OverlapTextGenerationPipeline,
)
from .pipeline_variants.text_generation import TextGenerationPipelineInterface
from .registry import PIPELINE_REGISTRY

__all__ = ["AutoModel", "AutoTokenizer"]

_logger = logging.getLogger("max.pipelines")

_DEBUG_DEFAULTS: dict[str, object] = {
    "enable_overlap_scheduler": False,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
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


def _config_sections(config: PipelineConfig) -> tuple[BaseModel, ...]:
    return (
        config,
        config.runtime,
        config.sampling,
        config.model,
        config.model.kv_cache,
    )


def _config_fields(
    config: PipelineConfig, values: Mapping[str, object]
) -> dict[str, object]:
    """Returns the entries of ``values`` that name a field of ``config``."""
    return {
        name: value
        for name, value in values.items()
        if any(
            name in type(section).model_fields
            for section in _config_sections(config)
        )
    }


def _unmet_values(
    config: PipelineConfig, required: Mapping[str, object]
) -> dict[str, object]:
    return {
        name: value
        for name, value in required.items()
        for section in _config_sections(config)
        if name in type(section).model_fields
        and getattr(section, name) != value
    }


def _build_config(
    model_path: str,
    max_batch_size: int,
    device_graph_capture: bool,
    config_overrides: Mapping[str, object],
) -> PipelineConfig:
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
    if accelerator_api() == "hip":
        # Serve turns the vendor BLAS fallback off whenever it turns capture
        # on, so this compiles the AMD matmul kernels serve would.
        flat["disable_vendor_blas_fallback"] = True
    flat.update(config_overrides)
    pipeline_config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(model_path=model_path, **flat)
    )
    _check_supported(pipeline_config, "AutoModel")
    if (
        not pipeline_config.runtime.enable_overlap_scheduler
        or device_graph_capture
        or "force" in config_overrides
        or config_overrides.get("enable_overlap_scheduler", False) is not False
    ):
        return pipeline_config
    # Only force keeps an architecture's overlap auto-enable off, and force
    # also skips the architecture's required arguments.
    arch = PIPELINE_REGISTRY.architecture_for_config(
        pipeline_config, task=PipelineTask.TEXT_GENERATION
    )
    required = _config_fields(
        pipeline_config, arch.required_arguments if arch is not None else {}
    )
    pipeline_config = PipelineConfig.from_args(
        PipelineArgs.from_flat_kwargs(
            model_path=model_path, **{**flat, **required, "force": True}
        )
    )
    if unmet := _unmet_values(pipeline_config, required):
        raise ValueError(
            f"The architecture requires {unmet}, which the config overrides "
            "contradict."
        )
    return pipeline_config


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
    """Loads a text generation model in-process.

    Unlike Hugging Face's base-model ``AutoModel``, this one generates. The
    Hugging Face equivalent is ``AutoModelForCausalLM``. Serving features are
    tested through ``max serve``.

    The config turns off the overlap scheduler, device graph capture, prefix
    caching and chunked prefill. On AMD it also turns off the vendor BLAS
    fallback, as serve does with capture on, so matmuls compile to serve's
    kernels. Where an architecture turns overlap on by itself, the caller
    passes no ``force``, and ``enable_overlap_scheduler`` is unset or
    ``False``, AutoModel rebuilds the config with ``force=True`` and passes
    the architecture's ``required_arguments`` explicitly. As in serve, these
    required arguments win over flat overrides.

    References to the pipeline, its KV cache manager, an open
    :class:`~max.pipelines.lib.generate.Batch`, or a live traceback from
    inside the pipeline keep device memory alive past :meth:`close`.

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
        **config_overrides: Flat :class:`~max.pipelines.lib.PipelineArgs`
            field names, such as ``device_specs`` or
            ``enable_prefix_caching``. They win over the debug defaults.

    Raises:
        TypeError: If both ``devices`` and ``device_specs`` are passed, if an
            override passes a ``runtime``, ``kv_cache`` or
            ``model`` section, or a ``config_file``, instead of flat fields,
            or a ``model_override`` entry sets one of them whole.
        ValueError: If the config needs a feature that only ``max serve``
            runs, such as LoRA, if the model takes image inputs or does not
            generate text, if a
            ``model_override`` entry contradicts an argument the architecture
            requires, or if ``device_graph_capture=True`` builds a pipeline
            that cannot capture graphs.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_batch_size: int = 1,
        devices: str | None = None,
        device_graph_capture: bool = False,
        **config_overrides: object,
    ) -> None:
        if devices is not None:
            if "device_specs" in config_overrides:
                raise TypeError("Pass devices or device_specs, not both.")
            # The config parses the same strings as the CLI's --devices.
            config_overrides["device_specs"] = devices
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
        self._pipeline_config = _build_config(
            model_path, max_batch_size, device_graph_capture, config_overrides
        )
        _, pipeline = PIPELINE_REGISTRY.retrieve(self._pipeline_config)
        assert isinstance(pipeline, TextGenerationPipelineInterface), (
            f"{model_path} does not load as a text generation pipeline"
        )
        self._pipeline: TextGenerationPipelineInterface[TextContext] | None = (
            pipeline
        )
        if not device_graph_capture:
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
        """The loaded pipeline.

        Raises:
            RuntimeError: If the model has been closed.
        """
        if self._pipeline is None:
            raise RuntimeError("AutoModel is closed. Create a new one.")
        return self._pipeline

    @property
    def pipeline_config(self) -> PipelineConfig:
        """The resolved config that built the pipeline."""
        return self._pipeline_config

    def generate(
        self,
        input_ids: Iterable[Sequence[int]],
        attention_mask: Iterable[Sequence[int]] | None = None,
        *,
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
    ) -> GenerateResult:
        """Runs :func:`~max.pipelines.lib.generate.generate` on :attr:`pipeline`.

        Takes the same arguments, so ``model.generate(**tokenizer(prompts),
        max_new_tokens=16)`` works with an :class:`AutoTokenizer`.

        Raises:
            TypeError: As described for
                :func:`~max.pipelines.lib.generate.generate`.
            ValueError: As described for
                :func:`~max.pipelines.lib.generate.generate`.
            RuntimeError: If the model has been closed, or as described for
                :func:`~max.pipelines.lib.generate.generate`.
        """
        return generate(
            self.pipeline,
            input_ids,
            attention_mask,
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
        )

    def batch(
        self,
        input_ids: Iterable[Sequence[int]],
        attention_mask: Iterable[Sequence[int]] | None = None,
        *,
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
    ) -> Batch:
        """Opens a :class:`~max.pipelines.lib.generate.Batch` on :attr:`pipeline`.

        Takes the same keyword arguments. Use the batch as a context manager
        so its rows are released.

        Raises:
            TypeError: As described for
                :class:`~max.pipelines.lib.generate.Batch`.
            ValueError: As described for
                :class:`~max.pipelines.lib.generate.Batch`.
            RuntimeError: If the model has been closed, or as described for
                :class:`~max.pipelines.lib.generate.Batch`.
        """
        return Batch(
            self.pipeline,
            input_ids,
            attention_mask,
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
        )

    def close(self) -> None:
        """Shuts down the KV cache manager and releases the pipeline.

        Calling it again does nothing. The pipeline is released even if the
        shutdown raises.
        """
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
