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
"""``generate_images`` over a fake image pipeline and tokenizer.

The fake tokenizer records each request and builds a real ``PixelContext``
from it, and the fake pipeline draws each image's pixels from the request's
seed, so the tests can check the request mapping and the decoded pixels
without a checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pytest
from max.pipelines.architectures.wan.context import WanContext
from max.pipelines.context import (
    GenerationOutput,
    GenerationStatus,
    PixelContext,
    PromptTooLongError,
    TextContext,
    TokenBuffer,
)
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.generate_images import (
    generate_images,
)
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.registry import PIPELINE_REGISTRY
from max.pipelines.modeling.types import (
    PipelineOutputsDict,
    PixelGenerationInputs,
    RequestID,
)
from max.pipelines.request import (
    OpenResponsesRequest,
    OutputContent,
    OutputImageContent,
    OutputTextContent,
)
from max.pipelines.request.provider_options import ImageProviderOptions
from PIL import Image

MODEL_PATH = "org/image-model"
# The fake tokenizer's size when a request leaves it unset.
DEFAULT_SIZE = 16


@pytest.fixture(autouse=True)
def pixel_context_arch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolves the fake config to an image architecture."""
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: PixelContext,
    )


def _pixels(
    seed: int | None, index: int, height: int, width: int
) -> npt.NDArray[np.uint8]:
    rng = np.random.default_rng([seed or 0, index])
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


class FakeImageTokenizer:
    """Records each request and builds a context from its options."""

    def __init__(self) -> None:
        self.requests: list[OpenResponsesRequest] = []
        self.too_long: set[str] = set()

    async def new_context(
        self,
        request: OpenResponsesRequest,
        input_image: Image.Image | None = None,
    ) -> PixelContext:
        self.requests.append(request)
        if request.body.input in self.too_long:
            raise PromptTooLongError(600, 512)
        options = request.body.provider_options.image
        assert options is not None
        return PixelContext(
            request_id=request.request_id,
            tokens=TokenBuffer(np.array([1], dtype=np.int64)),
            height=options.height or DEFAULT_SIZE,
            width=options.width or DEFAULT_SIZE,
            num_images_per_prompt=options.num_images,
            output_format=options.output_format,
            seed=request.body.seed,
        )


class FakeImagePipeline:
    """Returns seeded pixels in the format each context asks for."""

    def __init__(self) -> None:
        config = MagicMock(spec=PipelineConfig)
        # A diffusion pipeline has components and no main model.
        config.models = ModelManifest(
            {
                "transformer": MagicMock(
                    model_path=MODEL_PATH, model_name=MODEL_PATH
                )
            }
        )
        self.pipeline_config: PipelineConfig = config
        self.batch_sizes: list[int] = []
        self.output_text = False

    def execute(
        self, inputs: PixelGenerationInputs[PixelContext]
    ) -> PipelineOutputsDict[GenerationOutput]:
        self.batch_sizes.append(len(inputs.batch))
        outputs: dict[RequestID, GenerationOutput] = {}
        for request_id, context in inputs.batch.items():
            output: list[OutputContent] = [
                OutputImageContent.from_numpy(
                    _pixels(context.seed, index, context.height, context.width),
                    format=context.output_format,
                )
                for index in range(context.num_images_per_prompt)
            ]
            if self.output_text:
                output = [OutputTextContent(text="no image")]
            outputs[request_id] = GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=output,
            )
        return outputs


@pytest.fixture
def pipeline() -> FakeImagePipeline:
    return FakeImagePipeline()


@pytest.fixture
def tokenizer() -> FakeImageTokenizer:
    return FakeImageTokenizer()


def _options(request: OpenResponsesRequest) -> ImageProviderOptions:
    options = request.body.provider_options.image
    assert options is not None
    return options


def test_maps_arguments_to_the_serve_request(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    generate_images(
        pipeline,
        tokenizer,
        "a red fox",
        negative_prompt="blurry",
        height=128,
        width=256,
        num_inference_steps=4,
        guidance_scale=2.0,
        true_cfg_scale=1.5,
        num_images_per_prompt=2,
        seed=7,
    )
    [request] = tokenizer.requests
    assert request.body.model == MODEL_PATH
    assert request.body.input == "a red fox"
    assert request.body.seed == 7
    assert _options(request) == ImageProviderOptions(
        negative_prompt="blurry",
        height=128,
        width=256,
        steps=4,
        guidance_scale=2.0,
        true_cfg_scale=1.5,
        num_images=2,
        output_format="png",
    )


def test_leaves_unset_knobs_to_the_defaults(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    generate_images(pipeline, tokenizer, "a red fox")
    options = _options(tokenizer.requests[0])
    assert options.model_fields_set == {"num_images", "output_format"}
    assert options == ImageProviderOptions(num_images=1, output_format="png")


def test_seeds_each_prompt_after_the_last(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    generate_images(pipeline, tokenizer, ["a", "b", "c"], seed=2**32 - 2)
    assert [request.body.seed for request in tokenizer.requests] == [
        2**32 - 2,
        2**32 - 1,
        0,
    ]
    assert [request.body.input for request in tokenizer.requests] == [
        "a",
        "b",
        "c",
    ]
    assert len({request.request_id for request in tokenizer.requests}) == 3


def test_returns_one_image_list_per_prompt(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    result = generate_images(
        pipeline,
        tokenizer,
        ["a", "b"],
        height=128,
        width=136,
        num_images_per_prompt=2,
        seed=3,
    )
    # One request per execute, as the pipeline takes.
    assert pipeline.batch_sizes == [1, 1]
    assert [len(row) for row in result.images] == [2, 2]
    for row, images in enumerate(result.images):
        for index, image in enumerate(images):
            assert image.size == (136, 128)
            assert image.mode == "RGB"
            np.testing.assert_array_equal(
                np.asarray(image), _pixels(3 + row, index, 128, 136)
            )


def test_negative_prompt_is_shared_or_one_per_prompt(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    generate_images(pipeline, tokenizer, ["a", "b"], negative_prompt="dark")
    generate_images(
        pipeline, tokenizer, ["a", "b"], negative_prompt=["dark", "bright"]
    )
    assert [_options(r).negative_prompt for r in tokenizer.requests] == [
        "dark",
        "dark",
        "dark",
        "bright",
    ]


Call = Callable[[FakeImagePipeline, FakeImageTokenizer], object]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        pytest.param(
            lambda p, t: generate_images(p, t, []),
            "prompt must hold at least one prompt.",
            id="no-prompts",
        ),
        pytest.param(
            lambda p, t: generate_images(
                p, t, ["a", "b"], negative_prompt=["x"]
            ),
            "negative_prompt must be one string or one per prompt, got 1 "
            "for 2 prompts.",
            id="negative-prompt-count",
        ),
        pytest.param(
            # The tokenizer would reject it only after earlier prompts ran.
            lambda p, t: generate_images(p, t, ["a", ""]),
            "Prompt 1 is empty.",
            id="empty-prompt",
        ),
        pytest.param(
            lambda p, t: generate_images(p, t, "a", num_inference_steps=0),
            "num_inference_steps must be at least 1, got 0.",
            id="no-steps",
        ),
        pytest.param(
            lambda p, t: generate_images(p, t, "a", num_images_per_prompt=0),
            "num_images_per_prompt must be at least 1, got 0.",
            id="no-images",
        ),
        pytest.param(
            lambda p, t: generate_images(p, t, "a", seed=-1),
            "seed must be in [0, 2**32), got -1.",
            id="negative-seed",
        ),
        pytest.param(
            lambda p, t: generate_images(p, t, "a", seed=2**32),
            "seed must be in [0, 2**32), got 4294967296.",
            id="seed-too-big",
        ),
        pytest.param(
            lambda p, t: generate_images(p, t, "a", height=100),
            "greater than or equal to 128",
            id="height-below-the-schema-floor",
        ),
        pytest.param(
            lambda p, t: generate_images(p, t, "a", width=129),
            "width must be a multiple of 8",
            id="width-off-the-grid",
        ),
    ],
)
def test_rejects_bad_arguments_before_running(
    pipeline: FakeImagePipeline,
    tokenizer: FakeImageTokenizer,
    call: Call,
    message: str,
) -> None:
    with pytest.raises(ValueError) as error:
        call(pipeline, tokenizer)
    assert message in str(error.value)
    assert tokenizer.requests == []
    assert pipeline.batch_sizes == []


def test_text_models_point_to_generate(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: FakeImagePipeline,
    tokenizer: FakeImageTokenizer,
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: TextContext,
    )
    with pytest.raises(TypeError) as error:
        generate_images(pipeline, tokenizer, "a red fox")
    assert str(error.value) == (
        "generate_images() runs image generation models, and this model "
        "uses TextContext. Use generate() for text models."
    )
    assert tokenizer.requests == []


def test_video_models_point_to_serve(
    monkeypatch: pytest.MonkeyPatch,
    pipeline: FakeImagePipeline,
    tokenizer: FakeImageTokenizer,
) -> None:
    monkeypatch.setattr(
        PIPELINE_REGISTRY,
        "retrieve_context_type",
        lambda pipeline_config: WanContext,
    )
    with pytest.raises(ValueError) as error:
        generate_images(pipeline, tokenizer, "a red fox", caller="AutoModel")
    assert str(error.value).splitlines() == [
        "AutoModel does not support video generation. The model uses "
        "WanContext, which holds video frames, and AutoModel returns images.",
        "AutoModel runs the pipeline directly, outside max serve. Test video "
        "generation with this max serve command instead.",
        f"  max serve --model-path {MODEL_PATH}",
        "Set MAX_SERVE_API_TYPES='[\"responses\"]' first, which turns on "
        "the /v1/responses route.",
    ]
    assert tokenizer.requests == []
    assert pipeline.batch_sizes == []


def test_rejects_output_without_image_data(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    pipeline.output_text = True
    with pytest.raises(RuntimeError, match="returned output_text content"):
        generate_images(pipeline, tokenizer, "a red fox")


def test_rejects_a_too_long_prompt_before_running(
    pipeline: FakeImagePipeline, tokenizer: FakeImageTokenizer
) -> None:
    tokenizer.too_long = {"b"}
    with pytest.raises(PromptTooLongError) as error:
        generate_images(pipeline, tokenizer, ["a", "b", "c"])
    assert str(error.value).startswith("Prompt 1: Prompt is too long: 600 ")
    assert pipeline.batch_sizes == []
