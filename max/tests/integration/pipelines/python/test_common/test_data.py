# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test data for pipeline evaluation.

Separates test data from business logic for better maintainability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MockTextGenerationRequest:
    """Request for text generation testing, supporting both text-only and multimodal inputs."""

    prompt: str
    """The text prompt to be processed by the model."""

    images: list[str]
    """List of image URLs or file paths. Empty for text-only requests."""

    @property
    def is_multimodal(self) -> bool:
        """Returns True if this request includes images."""
        return len(self.images) > 0

    @classmethod
    def text_only(cls, prompt: str) -> MockTextGenerationRequest:
        """Creates a text-only generation request."""
        return cls(prompt=prompt, images=[])

    @classmethod
    def with_images(
        cls, prompt: str, images: list[str]
    ) -> MockTextGenerationRequest:
        """Creates a multimodal generation request."""
        return cls(prompt=prompt, images=images)


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


MULTIMODAL_IMAGE = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"

PIXTRAL_PROMPT = "<s>[INST]Describe the images.\n[IMG][/INST]"
PIXTRAL_IMAGE = "https://picsum.photos/id/237/400/300"

INTERNVL_INSTRUCT_PROMPT = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|image|>\nDescribe the image; where are these people and what are they doing?<|im_end|>\n<|im_start|>assistant\n"
INTERNVL_INSTRUCT_IMAGE = "https://cdn-uploads.huggingface.co/production/uploads/632698f75cf955bfbbe727a9/L9E3YzvMo2B_XvloSx5ud.jpeg"

IDEFICS3_INSTRUCT_PROMPT = "<image> Describe the image:"
IDEFICS3_INSTRUCT_IMAGE = "https://cdn.britannica.com/47/195447-050-51296461/Masjed-e-Jomeh-Yazd-Iran.jpg"


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
        prompt=PIXTRAL_PROMPT, images=[PIXTRAL_IMAGE]
    )
]

INTERNVL_INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_images(
        prompt=INTERNVL_INSTRUCT_PROMPT,
        images=[INTERNVL_INSTRUCT_IMAGE],
    )
]

IDEFICS3_INSTRUCT_REQUESTS = [
    MockTextGenerationRequest.with_images(
        IDEFICS3_INSTRUCT_PROMPT, [MULTIMODAL_IMAGE]
    ),
    MockTextGenerationRequest.with_images(
        IDEFICS3_INSTRUCT_PROMPT, [IDEFICS3_INSTRUCT_IMAGE]
    ),
]
