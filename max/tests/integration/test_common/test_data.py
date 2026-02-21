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
"""Test data for pipeline evaluation.

Separates test data from business logic for better maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from max.interfaces import (
    ImageContentPart,
    RequestID,
    SamplingParams,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.interfaces.provider_options import (
    ImageProviderOptions,
    ProviderOptions,
)
from max.interfaces.request import (
    OpenResponsesRequest,
    OpenResponsesRequestBody,
)
from max.support import fetch_bytes_from_s3


@dataclass(frozen=True)
class MockTextGenerationRequest:
    """Request for text generation testing, supporting both text-only and multimodal inputs."""

    prompt: str
    """The text prompt to be processed by the model."""

    images: list[str]
    """List of image URLs or file paths. Empty for text-only requests."""

    messages: list[TextGenerationRequestMessage]
    """List of messages to be processed by the model. If this is provided, the
    prompt is used to identify the request, while the messages are processed by
    the model."""

    is_multimodal: bool

    model_name: str = ""

    @classmethod
    def text_only(cls, prompt: str) -> MockTextGenerationRequest:
        """Creates a text-only generation request."""
        return cls(prompt=prompt, images=[], messages=[], is_multimodal=False)

    @classmethod
    def with_images(
        cls,
        prompt: str,
        images: list[str],
        messages: list[dict[str, str | Any]] | None = None,
    ) -> MockTextGenerationRequest:
        """Creates a multimodal generation request.

        Images are embedded in message content with their URLs/paths for
        extraction later when converting to TextGenerationRequest.
        """
        if messages is None:
            proper_messages = [
                TextGenerationRequestMessage(
                    role="user",
                    content=[TextContentPart(text=prompt)]
                    + [ImageContentPart() for _ in images],
                )
            ]
        else:
            proper_messages = []
            for message in messages:
                if isinstance(message, dict):
                    proper_messages.append(
                        TextGenerationRequestMessage(**message)
                    )

        return cls(
            prompt=prompt,
            images=images,
            messages=proper_messages,
            is_multimodal=True,
        )

    @classmethod
    def with_messages(
        cls, prompt: str, messages: list[dict[str, Any]], is_multimodal: bool
    ) -> MockTextGenerationRequest:
        """Creates a generation request with messages.

        Note that the prompt still needs to be passed in since it is used to
        identify the request.
        """
        # Extract image URLs/paths from messages content

        return cls(
            prompt=prompt,
            images=[],
            messages=[
                cast(TextGenerationRequestMessage, message)
                for message in messages
            ],
            is_multimodal=is_multimodal,
        )

    def to_text_generation_request(
        self, request_id: RequestID, sampling_params: SamplingParams
    ) -> TextGenerationRequest:
        if self.messages:
            return TextGenerationRequest(
                request_id=request_id,
                model_name=self.model_name,
                sampling_params=sampling_params,
                messages=self.messages,
                images=[fetch_bytes_from_s3(img) for img in self.images],
            )
        else:
            return TextGenerationRequest(
                request_id=request_id,
                model_name=self.model_name,
                sampling_params=sampling_params,
                prompt=self.prompt,
            )


@dataclass(frozen=True)
class MockPixelGenerationRequest:
    """Request for pixel generation testing."""

    prompt: str
    """The text prompt for image generation."""

    secondary_prompt: str | None = None
    """Optional secondary text prompt for dual text encoders."""

    negative_prompt: str | None = None
    """Optional negative prompt to guide what NOT to generate."""

    secondary_negative_prompt: str | None = None
    """Optional secondary negative prompt."""

    height: int | None = None
    """Height of generated image in pixels."""

    width: int | None = None
    """Width of generated image in pixels."""

    num_inference_steps: int = 50
    """Number of denoising steps."""

    guidance_scale: float = 3.5
    """Guidance scale for classifier-free guidance."""

    true_cfg_scale: float = 1.0
    """True CFG scale."""

    seed: int | None = None
    """Random seed for reproducibility."""

    model_name: str = ""
    """Model name for the request."""

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        *,
        secondary_prompt: str | None = None,
        negative_prompt: str | None = None,
        secondary_negative_prompt: str | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        true_cfg_scale: float = 1.0,
        seed: int | None = None,
        model_name: str = "",
    ) -> MockPixelGenerationRequest:
        """Creates a pixel generation request from a prompt."""
        return cls(
            prompt=prompt,
            secondary_prompt=secondary_prompt,
            negative_prompt=negative_prompt,
            secondary_negative_prompt=secondary_negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            true_cfg_scale=true_cfg_scale,
            seed=seed,
            model_name=model_name,
        )

    def to_open_responses_request(
        self, request_id: RequestID, model_name: str | None = None
    ) -> OpenResponsesRequest:
        """Convert to an OpenResponsesRequest for pixel generation."""
        body = OpenResponsesRequestBody(
            model=model_name or self.model_name,
            input=self.prompt,
            seed=self.seed,
            provider_options=ProviderOptions(
                image=ImageProviderOptions(
                    negative_prompt=self.negative_prompt,
                    secondary_prompt=self.secondary_prompt,
                    secondary_negative_prompt=self.secondary_negative_prompt,
                    height=self.height,
                    width=self.width,
                    steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    true_cfg_scale=self.true_cfg_scale,
                )
            ),
        )
        return OpenResponsesRequest(request_id=request_id, body=body)


# Existing test data extracted from evaluate.py
LONG_TEXT_PROMPT = """One of the most important aspects of performance benchmarking when it pertains to comparison of different implementations is making sure comparisons are fair. This is a place where most discussions occur, as deviation from best practices can make one’s performance claims easy to dismiss. For faster results of a given implementation (the Mojo implementation in our case) to be meaningful, the comparison needs to be apples-to-apples.
    * Make sure you use equivalent optimization flags across implementations; even though flags (like -O3 in C) that enable multiple optimizations at once cannot always be equivalent to another language’s -O3, make sure you don’t compare something like a debug build with an implementation that uses the fast optimization flag.
    * Make sure that if one implementation has auto-vectorization or automatic multithreading enabled the same applies to all implementations to be compared (unless for a given language one of these performs worse when turned-on, in which case one could keep the fastest implementation for comparison purposes).
    * Use the latest (or best) combination of compilers, libraries, etc. — an older compiler version (for example) may perform better for whatever reason; however it should be considered sufficient to test with the latest stable version. One can test with older or experimental versions if they are so inclined.
    * Use the same input file (if applicable) or same input data. Avoid random data generation that may stress different code paths.
    * Use the same algorithm (if applicable) across all your implementations.
    * Use equivalent error testing as it applies to different domains’ best practices (e.g., input sanitizing, corner case testing).
    * Remove any unnecessary I/O (e.g., writing to file/screen for debug purposes) and keep only what is practically necessary — make sure you do so in a manner that code is not optimized out (see #6)!
    * Try to apply the same level of manual optimization (within reason) — if you write multi-threaded/vectorized code in Mojo, you should try to compare it to an equivalent implementation of the other language. There is a case to be made here, however, if the other language does not have such capabilities or they are so difficult to use that implementing them is beyond what one can reasonably do. This can highlight the programmability aspect of Mojo (or one language against another more generally), but this fact should be listed so that people can take the performance claims under this light."""
SHORT_TEXT_PROMPTS = (
    "def is_prime(x):\n",
    "The meaning of life is ",
    """Translate the English text to Italian.
    Text: Sometimes, I've believed as many as six impossible things before breakfast.
    Translation:""",
)
MULTIMODAL_PROMPT = (
    "<|image|><|begin_of_text|>If I had to write a haiku for this one"
)


MULTIMODAL_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/multimodal_image.jpg"

PIXTRAL_PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"
PIXTRAL_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the images."},
            {"type": "image"},
        ],
    }
]
PIXTRAL_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/pixtral_image.jpg"

INTERNVL_INSTRUCT_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|image|> Describe the image; where are these people and what are they doing?<|im_end|>\n<|im_start|>assistant\n"
INTERNVL_INSTRUCT_MESSAGES = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {
                "type": "text",
                "text": "Describe the image; where are these people and what are they doing?",
            },
        ],
    },
]
INTERNVL_INSTRUCT_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/internvl_instruct_image.jpg"

IDEFICS3_INSTRUCT_PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<image> Describe the image:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
IDEFICS3_INSTRUCT_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image:"},
        ],
    }
]
IDEFICS3_INSTRUCT_IMAGE = "s3://modular-bazel-artifacts-public/artifacts/model_testdata/idefics3_instruct_image.jpg"


DEFAULT_PROMPTS = [LONG_TEXT_PROMPT, *SHORT_TEXT_PROMPTS]
DEFAULT_TEXT_ONLY = [
    MockTextGenerationRequest.text_only(prompt=prompt)
    for prompt in DEFAULT_PROMPTS
]

DEFAULT_MULTIMODAL = [
    MockTextGenerationRequest.with_images(
        prompt=MULTIMODAL_PROMPT,
        images=[MULTIMODAL_IMAGE],
    )
]

PIXTRAL_REQUESTS = [
    MockTextGenerationRequest.with_images(
        prompt=PIXTRAL_PROMPT,
        images=[PIXTRAL_IMAGE],
        messages=PIXTRAL_MESSAGES,
    )
]

INTERNVL_INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_images(
        prompt=INTERNVL_INSTRUCT_PROMPT,
        images=[INTERNVL_INSTRUCT_IMAGE],
        messages=INTERNVL_INSTRUCT_MESSAGES,
    )
]

IDEFICS3_INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_images(
        IDEFICS3_INSTRUCT_PROMPT,
        [MULTIMODAL_IMAGE],
        messages=IDEFICS3_INSTRUCT_MESSAGES,
    ),
    MockTextGenerationRequest.with_images(
        IDEFICS3_INSTRUCT_PROMPT,
        [IDEFICS3_INSTRUCT_IMAGE],
        messages=IDEFICS3_INSTRUCT_MESSAGES,
    ),
]

# Default pixel generation prompts
DEFAULT_PIXEL_GENERATION_PROMPTS = [
    "photography, soft natural textures, highly realistic light, editorial, A black panther stalking through the dense undergrowth of an Indian jungle, early evening with shadows from tall trees, captured with low light photography, high ISO setting to highlight the panther's muscles in motion",
    "Dramatic news broadcast scene in a Teahupoʻo wave's where a cow surfing, mimicking pro surf rider poses. Yogis laugh and take pictures. The news banner reads: 'COW win Olympics!!'",
    "Full body shot of a handsome tattooed short dark haired man wearing a jean and a white tee-shirt in 'Chiaroscuro Chronicles', lost in a captivating, slate gray monochromatic realm of masterful lighting and careful shading, emphasizing the emotional depth of the narrative, abrasive authenticity, ambient occlusion",
    "A beautiful woman in a red dress walking down a street",
    'The image show the fourth elements, each one in a part of the picture, first part is at top left and show a splashing multicolor water text with many water reflections, the text is made of water, the water word is "WATER", the background is splashing water, the second part of the image is a top right and show a soil rounded text, the word made of soil is "EARTH", the background is planet earth, the third part of the image is at bottom left and show a cloud multicolor rounded text, the word is "AIR" made of colorfull cloud the background is a sunset, and the last part of the image in the bottom right shows a red fire rounded text made of lava, the colorfull big word made of fire is "FIRE", the background is the closeup eruptive sun',
]

DEFAULT_PIXEL_GENERATION = [
    MockPixelGenerationRequest.from_prompt(prompt=prompt, seed=42)
    for prompt in DEFAULT_PIXEL_GENERATION_PROMPTS
]
