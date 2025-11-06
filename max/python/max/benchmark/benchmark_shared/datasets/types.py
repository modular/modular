# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Literal

from PIL import Image
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import TypedDict


class DatasetMode(str, Enum):
    """Enumeration of supported dataset loading modes.

    This enum defines the different ways datasets can be loaded:
    - LOCAL: Load from a local file path (from environment variable or --dataset-path)
    - HUGGINGFACE: Load from HuggingFace Hub (default behavior)
    """

    LOCAL = "local"
    HUGGINGFACE = "huggingface"


class OpenAIImageURL(TypedDict):
    url: str


class OpenAIImage(TypedDict):
    type: Literal["image_url"]
    image_url: OpenAIImageURL


@dataclass
class SampledRequest:
    prompt_formatted: str | list[dict[str, Any]]
    prompt_len: int
    output_len: int | None
    encoded_images: list[OpenAIImage]
    ignore_eos: bool


MessageSource = Literal["user", "assistant"]


@dataclass
class ChatMessage:
    source: MessageSource
    content: str
    num_tokens: int


@dataclass
class ChatSession:
    id: int | None
    messages: Sequence[ChatMessage]


def estimate_num_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_chat_message(
    source: MessageSource,
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    num_tokens: int | None = None,
) -> ChatMessage:
    return ChatMessage(
        source,
        prompt,
        num_tokens or estimate_num_tokens(tokenizer, prompt),
    )


def encode_image(img: Image.Image) -> OpenAIImage:
    """
    Convert the given PIL.Image.Image to JPEG and encode in base64.
    Returns an openai API image_url content entry with the encoded string.
    """
    img_buffer = BytesIO()
    # Drop alpha channel and convert to jpeg
    img.convert("RGB").save(img_buffer, format="JPEG")
    # Encode in base64 and convert to str
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    # return openai-api dict
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
    }


def encode_image_from_file_path(file_path: str) -> OpenAIImage:
    """
    Read an image file as raw bytes and encode in base64 without any transformations.
    Preserves the exact original file data and determines MIME type from file extension.

    Args:
        file_path: Path to the image file

    Returns:
        OpenAI API image_url content entry with the encoded string

    Raises:
        ValueError: If the file extension is not supported
        FileNotFoundError: If the file does not exist
    """
    extension_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    # Check if extension is supported
    _, ext = os.path.splitext(file_path.lower())
    if ext not in extension_to_mime:
        supported_exts = ", ".join(extension_to_mime.keys())
        raise ValueError(
            f"Unsupported image file extension '{ext}'. "
            f"Supported extensions: {supported_exts}"
        )

    # Base64 encode file bytes
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{extension_to_mime[ext]};base64,{img_base64}"
        },
    }
