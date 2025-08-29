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
# File contains code from the vllm project
# https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py
# derived from commit: dd53c4b023056cda6174cc32dc3d31bc01e8646a
# used under the Apache 2 licence
# and the InfiniteBench project
# https://github.com/OpenBMB/InfiniteBench/blob/main/src/eval_utils.py
# derived from commit: 51d9b37b0f1790ead936df2243abbf7f0420e439
# used under the MIT licence

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, TypedDict

import msgspec
from PIL import Image
from transformers import PreTrainedTokenizerBase


class OpenAIImageURL(TypedDict):
    url: str


class OpenAIImage(TypedDict):
    type: Literal["image_url"]
    image_url: OpenAIImageURL


@dataclass
class SampledRequest:
    prompt_formatted: str
    prompt_len: int
    output_len: int | None
    encoded_img: OpenAIImage | None


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


# -----------------------------------------------------------------------------
# Longcontext Dataset (code_debug)
# -----------------------------------------------------------------------------


CODE_DEBUG_TEMPLATE = """\
There is ONLY ONE function in the large project that is deliberately made to \
include an obvious error. Please find the function that contains the most \
obvious errors. I will give you four options to narrow your scope. You can \
inspect the options and think. Eventually, tell me the answer using one \
single letter (A, B, C, or D).

{context}

Which function has deliberate error?
A. {OPTION_A}
B. {OPTION_B}
C. {OPTION_C}
D. {OPTION_D}

You should first find the functions in the options. Repeat their content, \
inspect through code, and at last give me your answer for the function that \
has the deliberate and obvious error in A, B, C, or D.\
"""


class CodeDebugLine(msgspec.Struct):
    id: int
    context: str
    input: str
    answer: Sequence[str]
    options: Sequence[str]


class ObfuscatedConversationsLine(msgspec.Struct):
    timestamp: str
    conversation_id: str
    messages: str


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


# -----------------------------------------------------------------------------
# Multi-turn chat
# -----------------------------------------------------------------------------


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
