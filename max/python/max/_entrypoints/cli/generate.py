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

"""Utilities for generating text and images in the cli."""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
from max.driver import accelerator_api
from max.pipelines import (
    PIPELINE_REGISTRY,
    TextAndVisionTokenizer,
    TextGenerationPipelineInterface,
    TextTokenizer,
)
from max.pipelines.context import (
    LogitsProcessor,
    ProcessorInputs,
    SamplingParams,
    TextContext,
)
from max.pipelines.lib import (
    OverlapTextGenerationPipeline,
    PipelineArgs,
    PipelineConfig,
    PixelGenerationPipeline,
    PixelGenerationTokenizer,
)
from max.pipelines.lib.generate import BatchPipeline, GenerateIterator
from max.pipelines.lib.generate_images import (
    _is_image_generation_model,
    _prepare_requests,
    _run_requests,
)
from max.pipelines.logging_utils import log_basic_config
from max.pipelines.modeling.types import (
    ImageContentPart,
    Pipeline,
    PipelineTask,
    PipelineTokenizer,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.profiler import OneShotCapture, Tracer
from PIL import Image

from .metrics import TextGenerationMetrics

logger = logging.getLogger("max._entrypoints")

MODEL_NAME = "model"


class TrackMetrics:
    def __init__(self, metrics: TextGenerationMetrics):
        self.metrics = metrics
        self.first_token = True

    def __call__(self, inputs: ProcessorInputs) -> None:
        if self.first_token:
            self.first_token = False
            self.metrics.signpost("first_token")
            self.metrics.prompt_size = len(inputs.context.tokens)
        self.metrics.new_token()


async def stream_context(
    pipeline: BatchPipeline,
    context: TextContext,
    on_tokens: Callable[[list[int]], Awaitable[None]],
) -> None:
    """Runs ``context`` as a batch of one, awaiting ``on_tokens`` per step.

    Yields to the event loop after every step, so a SIGINT's cancellation
    lands before the next step and the batch releases the row.
    """
    with GenerateIterator.from_contexts(
        pipeline, [context], caller="max generate"
    ) as batch:
        while not batch.done:
            tokens = batch.step().get(0)
            if tokens:
                await on_tokens(tokens)
            await asyncio.sleep(0)


async def stream_text_to_console(
    pipeline: Pipeline[Any, Any],
    tokenizer: PipelineTokenizer[TextContext, Any, TextGenerationRequest],
    prompt: str,
    images: list[bytes],
    sampling_params: SamplingParams,
    metrics: TextGenerationMetrics | None = None,
    print_tokens: bool = True,
) -> None:
    assert isinstance(tokenizer, TextTokenizer | TextAndVisionTokenizer)
    logits_processors: list[LogitsProcessor] = []
    if metrics:
        logits_processors.append(TrackMetrics(metrics))

    sampling_params = dataclasses.replace(
        sampling_params, logits_processors=logits_processors
    )

    # Base/completion models (e.g. GPT-2) have no chat template. For these, send the raw
    # prompt so the tokenizer encodes it directly instead of trying to apply a
    # chat template. Chat models (with a template) and multimodal requests
    # still go through the messages path.
    # Custom tokenizers (--custom-architectures) may have no HF delegate at
    # all; treat them like template-less tokenizers.
    has_chat_template = bool(
        getattr(getattr(tokenizer, "delegate", None), "chat_template", None)
    )
    if images or has_chat_template:
        request = TextGenerationRequest(
            request_id=RequestID(),
            messages=[
                TextGenerationRequestMessage(
                    role="user",
                    content=[
                        TextContentPart(text=prompt),
                        *(ImageContentPart() for _ in images),
                    ],
                )
            ],
            images=images,
            model_name=MODEL_NAME,
            sampling_params=sampling_params,
        )
    else:
        request = TextGenerationRequest(
            request_id=RequestID(),
            prompt=prompt,
            model_name=MODEL_NAME,
            sampling_params=sampling_params,
        )

    async def print_decoded(tokens: list[int]) -> None:
        if print_tokens:
            decoded = await tokenizer.decode(tokens, skip_special_tokens=True)
            print(decoded, end="", flush=True)

    if metrics:
        metrics.signpost("begin_generation")
    try:
        assert isinstance(pipeline, TextGenerationPipelineInterface)
        context = await pipeline.tokenizer.new_context(request)
        await stream_context(pipeline, context, print_decoded)
    finally:
        if metrics:
            metrics.signpost("end_generation")


def _with_debug_defaults(pipeline_args: PipelineArgs) -> PipelineArgs:
    """Returns ``pipeline_args`` with the generate defaults for unset fields.

    Like transformers, ``max generate`` runs every step normally unless the
    caller passes ``--device-graph-capture``. Serve compiles AMD matmuls
    without the vendor BLAS fallback whenever it turns capture on, so on AMD
    this also turns the fallback off, to compile the kernels serve would. It
    runs one prompt at a time, so it plans for a batch of one, which also
    keeps graph capture warmup to a single graph.
    """
    runtime = pipeline_args.runtime
    defaults: dict[str, object] = {}
    if runtime.device_graph_capture is None:
        defaults["device_graph_capture"] = False
        if (
            accelerator_api() == "hip"
            and "disable_vendor_blas_fallback" not in runtime.model_fields_set
        ):
            defaults["disable_vendor_blas_fallback"] = True
    if runtime.max_batch_size is None:
        defaults["max_batch_size"] = 1
    if not defaults:
        return pipeline_args
    return pipeline_args.model_copy(
        update={"runtime": runtime.model_copy(update=defaults)}
    )


def generate_text_for_pipeline(
    pipeline_args: PipelineArgs,
    sampling_params: SamplingParams,
    prompt: str,
    image_urls: Iterable[str] = (),
    num_warmups: int = 0,
    profile: bool = False,
    profile_top_n: int = 15,
) -> None:
    # The capture handle is created outside `with TextGenerationMetrics` so
    # ``end_and_finalize`` can fire *after* the metrics report prints. Under
    # nsys, ``cudaProfilerStop`` triggers the ``.nsys-rep`` write — delaying
    # it past the metrics report keeps the normal generate output (text +
    # stats) from being buried inside nsys's file-writing progress lines.
    pipeline_config = PipelineConfig.from_args(
        _with_debug_defaults(pipeline_args)
    )
    retrieved = PIPELINE_REGISTRY.retrieve_factory(pipeline_config)
    tokenizer = retrieved.tokenizer
    pipeline = retrieved.factory()
    assert isinstance(pipeline, Pipeline)
    # Without warmup there are no captured graphs, and every step would run
    # eagerly despite the flag.
    if pipeline_config.runtime.device_graph_capture and isinstance(
        pipeline, OverlapTextGenerationPipeline
    ):
        logger.info("Warming up device graph capture")
        pipeline.warmup_graph_capture()
    log_basic_config(pipeline_config, memory_plan=retrieved.memory_plan)
    logger.info("max_batch_size: %d", pipeline.max_batch_size)

    capture = OneShotCapture(top_n=profile_top_n) if profile else None
    try:
        # Run timed run & print results.
        with TextGenerationMetrics(print_report=True) as metrics:
            if image_urls:
                logger.info("Downloading images")
                images = [requests.get(url).content for url in image_urls]
            else:
                images = []

            if num_warmups > 0:
                logger.info("Running warmup")
                warmup_params = dataclasses.replace(
                    sampling_params, max_new_tokens=num_warmups
                )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=warmup_params,
                        metrics=None,
                        print_tokens=False,
                    )
                )

            # Run and print results.
            logger.info("Beginning text generation")

            with contextlib.ExitStack() as exit_stack:
                if capture is not None:
                    capture.start()
                    # ``Tracer`` adds an NVTX "inference" label visible in
                    # nsys-ui. It must be entered after ``capture.start``
                    # so its NVTX range sits inside the cuda profiler
                    # window.
                    exit_stack.enter_context(
                        Tracer("inference", color="modular_purple")
                    )
                asyncio.run(
                    stream_text_to_console(
                        pipeline,
                        tokenizer,
                        prompt,
                        images,
                        sampling_params=sampling_params,
                        metrics=metrics,
                        print_tokens=True,
                    )
                )
        # ``TextGenerationMetrics.__exit__`` has printed the report here.
        # End the capture *now*, so nsys's file-writing output follows.
    finally:
        if capture is not None:
            capture.end_and_finalize()


# The flags only one kind of model reads, by click parameter name.
_TEXT_ONLY_FLAGS = (
    "max_new_tokens",
    "min_new_tokens",
    "top_k",
    "top_p",
    "min_p",
    "temperature",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "ignore_eos",
    "stop",
    "stop_token_ids",
    "detokenize",
    "num_warmups",
    "profile",
    "profile_output",
    "profile_top_n",
    # An image pipeline runs one prompt per call and captures no graphs, so
    # these are rejected as AutoModel rejects them.
    "device_graph_capture",
    "max_batch_size",
)
_IMAGE_ONLY_FLAGS = (
    "negative_prompt",
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "true_cfg_scale",
    "strength",
    "num_images_per_prompt",
    "output",
)


def check_flags_fit_model(
    set_flags: Mapping[str, str], image_model: bool, model_path: str
) -> None:
    """Raises if the user set a flag only the other kind of model reads.

    Args:
        set_flags: The flag the user set for each click parameter name.
        image_model: Whether the model generates images.
        model_path: The model, which the error names.

    Raises:
        ValueError: Naming every flag for the other kind of model.
    """
    wrong = _TEXT_ONLY_FLAGS if image_model else _IMAGE_ONLY_FLAGS
    flags = [set_flags[name] for name in wrong if name in set_flags]
    if not flags:
        return
    kind, loaded = ("text", "an image") if image_model else ("image", "a text")
    verb = "applies" if len(flags) == 1 else "apply"
    raise ValueError(
        f"{', '.join(flags)} {verb} to {kind} generation models, and "
        f"{model_path} is {loaded} generation model."
    )


def is_image_generation_model(pipeline_args: PipelineArgs) -> bool:
    """Returns whether ``pipeline_args`` names an image generation model.

    It resolves the task before anything loads, with the lookup
    :class:`~max.pipelines.AutoModel` uses.

    Raises:
        ValueError: If ``--task pixel_generation`` names a model that does
            not generate images.
    """
    task = pipeline_args.task
    if task not in (PipelineTask.UNDEFINED, PipelineTask.PIXEL_GENERATION):
        return False
    if _is_image_generation_model(pipeline_args):
        return True
    if task == PipelineTask.PIXEL_GENERATION:
        raise ValueError(
            "--task pixel_generation needs an image generation model, but "
            f"{pipeline_args.model_path} uses "
            f"{pipeline_args.main_architecture_name}, which is not one."
        )
    return False


def _image_paths(output: Path, count: int) -> list[Path]:
    """Returns where to save ``count`` images, checking ``output`` first.

    A path without a suffix is a directory, created if missing.

    Raises:
        ValueError: If the suffix of ``output`` names no format PIL writes or
            its directory does not exist.
    """
    if not output.suffix:
        output.mkdir(parents=True, exist_ok=True)
        output = output / "image.png"
    image_format = Image.registered_extensions().get(output.suffix.lower())
    if image_format not in Image.SAVE:
        raise ValueError(
            f"--output {output} has the suffix {output.suffix}, which names "
            "no image format PIL can write. Use one such as .png or .jpg."
        )
    if not output.parent.is_dir():
        raise ValueError(
            f"--output {output} is in {output.parent}, which does not exist."
        )
    if count == 1:
        return [output]
    return [
        output.with_name(f"{output.stem}-{index}{output.suffix}")
        for index in range(count)
    ]


def generate_images_for_pipeline(
    pipeline_args: PipelineArgs,
    prompt: str,
    image_urls: Sequence[str],
    output: Path,
    *,
    negative_prompt: str | None,
    height: int | None,
    width: int | None,
    num_inference_steps: int | None,
    guidance_scale: float | None,
    true_cfg_scale: float | None,
    strength: float | None,
    num_images_per_prompt: int,
    seed: int,
) -> list[Path]:
    """Generates images for ``prompt`` and saves them under ``output``.

    The config is built as serve builds it, without ``max generate``'s text
    defaults.

    Returns:
        The saved paths. The suffix of ``output`` picks the format, and a
        path without one is a directory that gets ``image.png``.

    Raises:
        ValueError: If there is more than one image URL, ``output`` is not a
            path an image can be saved to, the model generates video, or as
            raised by
            :func:`~max.pipelines.lib.generate_images.generate_images`. All
            of these come before the model loads.
    """
    if len(image_urls) > 1:
        raise ValueError(
            "An image generation model takes one --image_url, the input "
            f"image, but got {len(image_urls)}."
        )
    paths = _image_paths(output, num_images_per_prompt)
    pipeline_config = PipelineConfig.from_args(pipeline_args)
    images = (
        [[requests.get(url).content for url in image_urls]]
        if image_urls
        else None
    )
    prepared = _prepare_requests(
        pipeline_config,
        prompt,
        images=images,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        true_cfg_scale=true_cfg_scale,
        strength=strength,
        num_images_per_prompt=num_images_per_prompt,
        seed=seed,
        caller="max generate",
    )
    tokenizer, pipeline = PIPELINE_REGISTRY.retrieve(
        pipeline_config, task=PipelineTask.PIXEL_GENERATION
    )
    assert isinstance(pipeline, PixelGenerationPipeline)
    assert isinstance(tokenizer, PixelGenerationTokenizer)
    [prompt_images] = _run_requests(pipeline, tokenizer, prepared).images
    for image, path in zip(prompt_images, paths, strict=True):
        image.save(path)
        print(f"Saved {path}")
    return paths
