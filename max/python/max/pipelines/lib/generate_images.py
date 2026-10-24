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

"""Image generation from text prompts, run in the caller's process.

Each prompt becomes the same request ``max serve`` builds for
``POST /v1/responses``, so the model's defaults and the request's validation
are the same as in serve. The images come back as PNG, so they match
serve's images for ``output_format="png"``. Serve defaults to JPEG.
Prompts run one at a time, since an image pipeline takes one request per
``execute``.
"""

from __future__ import annotations

import base64
import dataclasses
import io
from collections.abc import Mapping, Sequence
from typing import Protocol

from max.pipelines.context import GenerationOutput, PixelContext, TextContext
from max.pipelines.modeling.types import (
    PipelineOutputsDict,
    PipelineTask,
    PixelGenerationInputs,
    RequestID,
)
from max.pipelines.request import (
    OpenResponsesRequest,
    OpenResponsesRequestBody,
    OutputImageContent,
)
from max.pipelines.request.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from PIL import Image

from .config import PipelineArgs, PipelineConfig
from .generate import _holds_video, _run_coroutine, _unsupported
from .registry import PIPELINE_REGISTRY

__all__ = [
    "GenerateImagesResult",
    "ImagePipeline",
    "ImageTokenizer",
    "generate_images",
]

# numpy's RandomState, which draws the initial noise, takes 32-bit seeds.
_SEED_LIMIT = 1 << 32


class ImagePipeline(Protocol):
    """The pipeline surface :func:`generate_images` drives.

    :class:`~max.pipelines.PixelGenerationPipeline` satisfies it.
    """

    @property
    def pipeline_config(self) -> PipelineConfig:
        """The config that names the model and its architecture."""
        ...

    def execute(
        self, inputs: PixelGenerationInputs[PixelContext]
    ) -> PipelineOutputsDict[GenerationOutput]:
        """Runs one request through the whole denoising loop."""
        ...


class ImageTokenizer(Protocol):
    """The tokenizer surface :func:`generate_images` drives.

    :class:`~max.pipelines.lib.PixelGenerationTokenizer` satisfies it.
    """

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: Image.Image | None = None,
    ) -> PixelContext:
        """Builds the context, with its initial noise, for ``request``."""
        ...


@dataclasses.dataclass(frozen=True)
class GenerateImagesResult:
    """The result of :func:`generate_images`."""

    images: list[list[Image.Image]]
    """One list per prompt, in prompt order, so ``images[i]`` holds prompt
    ``i``'s ``num_images_per_prompt`` images."""


def _per_prompt(
    value: str | Sequence[str] | None, num_prompts: int, name: str
) -> list[str | None]:
    if value is None or isinstance(value, str):
        return [value] * num_prompts
    values: list[str | None] = list(value)
    if len(values) != num_prompts:
        raise ValueError(
            f"{name} must be one string or one per prompt, got "
            f"{len(values)} for {num_prompts} prompts."
        )
    return values


def _is_image_generation_model(pipeline_args: PipelineArgs) -> bool:
    """Returns whether ``pipeline_args`` names a pixel generation model.

    Image and video architectures both count. It needs no :class:`PipelineConfig`,
    so it runs before one is built.
    """
    # A custom architecture that overrides a built-in one has to register
    # before the lookup reads the registry, as serve does.
    PIPELINE_REGISTRY._import_custom_architectures(
        pipeline_args.runtime.custom_architectures
    )
    try:
        name = pipeline_args.main_architecture_name
    except ValueError:
        # Leave a missing architecture to the text path.
        return False
    # The lookup falls back to an architecture of another task, so the task
    # it found decides.
    arch = PIPELINE_REGISTRY.retrieve_architecture(
        name,
        prefer_module_v3=pipeline_args.runtime.prefer_module_v3,
        task=PipelineTask.PIXEL_GENERATION,
    )
    return arch is not None and arch.task == PipelineTask.PIXEL_GENERATION


def _check_image_model(pipeline_config: PipelineConfig, caller: str) -> None:
    context_type = PIPELINE_REGISTRY.retrieve_context_type(pipeline_config)
    if not issubclass(context_type, PixelContext):
        hint = (
            " Use generate() for text models."
            if issubclass(context_type, TextContext)
            else ""
        )
        raise TypeError(
            f"{caller} runs image generation models, and this model uses "
            f"{context_type.__name__}.{hint}"
        )
    if _holds_video(context_type):
        raise _unsupported(
            pipeline_config,
            "video generation",
            f"The model uses {context_type.__name__}, which holds video "
            f"frames, and {caller} returns images",
            caller=caller,
            note=(
                "Set MAX_SERVE_API_TYPES='[\"responses\"]' first, which "
                "turns on the /v1/responses route."
            ),
        )


def _image_options(
    knobs: Mapping[str, object], num_images: int
) -> ImageProviderOptions:
    # Unset knobs keep the schema's and the tokenizer's defaults, as in serve.
    options = {
        name: value for name, value in knobs.items() if value is not None
    }
    # PNG keeps every pixel the model produced.
    return ImageProviderOptions.model_validate(
        {**options, "num_images": num_images, "output_format": "png"}
    )


def _decode(output: GenerationOutput) -> list[Image.Image]:
    images: list[Image.Image] = []
    for content in output.output:
        if (
            not isinstance(content, OutputImageContent)
            or content.image_data is None
        ):
            raise RuntimeError(
                f"The pipeline returned {content.type} content without "
                "image data."
            )
        image = Image.open(io.BytesIO(base64.b64decode(content.image_data)))
        image.load()
        images.append(image)
    return images


async def _new_contexts(
    tokenizer: ImageTokenizer, requests: Sequence[OpenResponsesRequest]
) -> list[PixelContext]:
    contexts = []
    for index, request in enumerate(requests):
        try:
            contexts.append(await tokenizer.new_context(request))
        except ValueError as error:
            error.args = (f"Prompt {index}: {error}",)
            raise
    return contexts


def generate_images(
    pipeline: ImagePipeline,
    tokenizer: ImageTokenizer,
    prompt: str | Sequence[str],
    *,
    negative_prompt: str | Sequence[str] | None = None,
    height: int | None = None,
    width: int | None = None,
    num_inference_steps: int | None = None,
    guidance_scale: float | None = None,
    true_cfg_scale: float | None = None,
    num_images_per_prompt: int = 1,
    seed: int = 0,
    caller: str = "generate_images()",
) -> GenerateImagesResult:
    """Generates images for each prompt, one prompt per pipeline run.

    An argument left as ``None`` takes the model's default, as a ``max
    serve`` request that omits it does.

    Args:
        pipeline: The image pipeline to run.
        tokenizer: The pipeline's tokenizer, which builds each request's
            context.
        prompt: One prompt, or a list of prompts.
        negative_prompt: What to steer away from, as one string for every
            prompt or one per prompt.
        height: The image height in pixels.
        width: The image width in pixels.
        num_inference_steps: The number of denoising steps.
        guidance_scale: How closely the image follows the prompt.
        true_cfg_scale: The true classifier-free guidance scale. Models
            that use it need it above 1 and a ``negative_prompt``.
        num_images_per_prompt: The number of images for each prompt.
        seed: The seed for prompt 0. Prompt ``i`` uses
            ``(seed + i) % 2**32``, since numpy takes 32-bit seeds.
        caller: What the user called, which errors name.

    Returns:
        One list of PIL images per prompt, in prompt order.

    Raises:
        TypeError: If the pipeline's model does not generate images.
        ValueError: If there are no prompts, a prompt is empty or too long
            for the model, ``negative_prompt`` has the wrong length,
            ``num_inference_steps`` or ``num_images_per_prompt`` is below 1,
            ``seed`` is outside ``[0, 2**32)``, a value is outside the range
            ``max serve`` accepts, or the model generates video. A
            tokenizer's error names its prompt. Every prompt is checked
            before the first one runs.
        RuntimeError: If the pipeline returns output that is not an image.
    """
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    if not prompts:
        raise ValueError("prompt must hold at least one prompt.")
    if "" in prompts:
        raise ValueError(f"Prompt {prompts.index('')} is empty.")
    negative_prompts = _per_prompt(
        negative_prompt, len(prompts), "negative_prompt"
    )
    if num_inference_steps is not None and num_inference_steps < 1:
        raise ValueError(
            "num_inference_steps must be at least 1, got "
            f"{num_inference_steps}."
        )
    if num_images_per_prompt < 1:
        raise ValueError(
            "num_images_per_prompt must be at least 1, got "
            f"{num_images_per_prompt}."
        )
    if not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"seed must be in [0, 2**32), got {seed}.")
    pipeline_config = pipeline.pipeline_config
    _check_image_model(pipeline_config, caller)

    knobs: dict[str, object] = {
        "height": height,
        "width": width,
        "steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "true_cfg_scale": true_cfg_scale,
    }
    model_name = pipeline_config.models.model_name
    requests = [
        OpenResponsesRequest(
            request_id=RequestID(),
            body=OpenResponsesRequestBody(
                model=model_name,
                input=row_prompt,
                seed=(seed + row) % _SEED_LIMIT,
                provider_options=ProviderOptions(
                    image=_image_options(
                        {**knobs, "negative_prompt": row_negative_prompt},
                        num_images_per_prompt,
                    )
                ),
            ),
        )
        for row, (row_prompt, row_negative_prompt) in enumerate(
            zip(prompts, negative_prompts, strict=True)
        )
    ]

    # Every context is built first, so a prompt the tokenizer rejects fails
    # before any prompt runs.
    contexts = _run_coroutine(_new_contexts(tokenizer, requests))
    images: list[list[Image.Image]] = []
    for context in contexts:
        outputs = pipeline.execute(
            PixelGenerationInputs(batch={context.request_id: context})
        )
        images.append(_decode(outputs[context.request_id]))
    return GenerateImagesResult(images=images)
