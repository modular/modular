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

from typing import Any

from PIL import Image

# docstyle-ignore
"""
These system prompts come from:
https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/system_messages.py#L54
"""

# docstyle-ignore
SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""


def format_input(
    prompts: list[str],
    system_message: str = SYSTEM_MESSAGE,
    images: list[Image.Image] | list[list[Image.Image]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Format a batch of text prompts into the conversation format expected by apply_chat_template.

    Optionally, add images to the input.

    Adapted from:
    https://github.com/black-forest-labs/flux2/blob/5a5d316b1b42f6b59a8c9194b77c8256be848432/src/flux2/text_encoder.py#L68

    Args:
        prompts: List of text prompts.
        system_message: System message to use (default: SYSTEM_MESSAGE).
        images: Optional list of images to add to the input.

    Returns:
        List of conversations, where each conversation is a list of message dicts.
    """
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    if images is None or len(images) == 0:
        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]
    else:
        assert len(images) == len(prompts), (
            "Number of images must match number of prompts"
        )
        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
            ]
            for _ in cleaned_txt
        ]

        for i, (el, img_list) in enumerate(zip(messages, images, strict=False)):
            # optionally add the images per batch element.
            if img_list is not None:
                # Handle both single images and lists of images
                img_items = (
                    img_list if isinstance(img_list, list) else [img_list]
                )
                el.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_obj}
                            for image_obj in img_items
                        ],
                    }
                )
            # add the text.
            el.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": cleaned_txt[i]}],
                }
            )

        return messages


def format_input_klein(
    prompts: list[str],
    images: list[Image.Image] | list[list[Image.Image]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Format prompts for Flux2 Klein (Qwen3): user-only plain-string content."""
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]
    if images is None or len(images) == 0:
        return [[{"role": "user", "content": prompt}] for prompt in cleaned_txt]

    assert len(images) == len(prompts), (
        "Number of images must match number of prompts"
    )
    return [
        [{"role": "user", "content": cleaned_txt[i]}]
        for i in range(len(cleaned_txt))
    ]
