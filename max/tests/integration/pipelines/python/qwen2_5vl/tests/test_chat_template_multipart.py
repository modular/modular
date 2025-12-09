# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Test that chat template properly handles multi-part content with images and text."""

import base64
from io import BytesIO

import numpy as np
from max.interfaces.pipeline_variants.text_generation import (
    TextGenerationRequestMessage,
)
from max.pipelines.architectures.qwen2_5vl.tokenizer import (
    Qwen2_5VLTokenizer,
)
from PIL import Image


def test_image(width: int = 224, height: int = 224) -> Image.Image:
    """Creates a test image for testing."""
    img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def image_to_base64(image: Image.Image) -> str:
    """Converts a PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def test_chat_template_preserves_text_with_image() -> None:
    """Test that apply_chat_template preserves text content when image is present.

    This is a regression test for PAQ-1381 where text content was being dropped
    when processing multi-part content containing both images and text.
    """
    # Create tokenizer.
    tokenizer = Qwen2_5VLTokenizer(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    # Create a test image.
    image = test_image(224, 224)
    image_b64 = image_to_base64(image)

    # Create message structure that mimics OpenAI API format.
    test_question = (
        "When a spring does work on an object, we cannot find the work by "
        "simply multiplying the spring force by the object's displacement. "
        "What is the compression distance?"
    )

    messages: list[TextGenerationRequestMessage] = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_b64,
                },
                {
                    "type": "text",
                    "text": test_question,
                },
            ],
        },
    ]

    # Apply chat template.
    prompt = tokenizer.apply_chat_template(messages)

    # The text content must be present in the prompt.
    assert test_question in prompt, (
        f"Text content was dropped from chat template output!\n"
        f"Expected to find: {test_question}\n"
        f"But prompt was: {prompt}"
    )

    # Check to ensure structure is correct.
    assert "<|im_start|>system" in prompt, "Missing system message marker"
    assert "You are a helpful assistant." in prompt, (
        "Missing system message content"
    )
    assert "<|im_start|>user" in prompt, "Missing user message marker"
    assert "<|vision_start|>" in prompt, "Missing vision start marker"
    assert "<|image_pad|>" in prompt, "Missing image placeholder"
    assert "<|vision_end|>" in prompt, "Missing vision end marker"

    # The question should come after the vision tokens.
    vision_end_idx = prompt.index("<|vision_end|>")
    question_idx = prompt.index(test_question)
    assert question_idx > vision_end_idx, (
        "Text content should appear after vision tokens, but found in wrong order"
    )


def test_chat_template_multiple_text_parts() -> None:
    """Test that chat template handles multiple text parts in content."""
    tokenizer = Qwen2_5VLTokenizer(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    image = test_image(224, 224)
    image_b64 = image_to_base64(image)

    text_part_1 = "First part of the question."
    text_part_2 = "Second part of the question."

    messages: list[TextGenerationRequestMessage] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_part_1,
                },
                {
                    "type": "image",
                    "image": image_b64,
                },
                {
                    "type": "text",
                    "text": text_part_2,
                },
            ],
        },
    ]

    prompt = tokenizer.apply_chat_template(messages)

    # Both text parts must be present.
    assert text_part_1 in prompt, (
        f"First text part missing from prompt: {prompt}"
    )
    assert text_part_2 in prompt, (
        f"Second text part missing from prompt: {prompt}"
    )


def test_chat_template_text_only() -> None:
    """Test that chat template still works correctly for text-only messages."""
    tokenizer = Qwen2_5VLTokenizer(
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
    )

    test_question = "What is 2 + 2?"

    messages: list[TextGenerationRequestMessage] = [
        {
            "role": "user",
            "content": test_question,
        },
    ]

    prompt = tokenizer.apply_chat_template(messages)

    # Text must be present for simple text-only case.
    assert test_question in prompt, (
        f"Text content missing from prompt: {prompt}"
    )
    assert "<|im_start|>user" in prompt, "Missing user message marker"
