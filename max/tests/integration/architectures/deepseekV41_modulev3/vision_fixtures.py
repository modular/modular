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
"""Configs, tokenizers and requests the image path's tests share."""

from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import NonCallableMock

import abi_contract
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    IMAGE_PLACEHOLDER,
)
from max.pipelines.architectures.deepseekV41_modulev3.registration import (
    RegisteredDeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41_modulev3.vision_encoder import (
    DeepseekV41VisionConfig,
)
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from PIL import Image
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

IMAGE = 129264
"""The checkpoint's ``image_token_id``."""


def vision_config() -> DeepseekV41VisionConfig:
    """The full checkpoint's vision encoder config."""
    full = abi_contract.hf_config("full")
    return DeepseekV41VisionConfig.from_hf(
        full["vision_config"], full["text_config"]["hidden_size"]
    )


def registered_config(**fields: Any) -> RegisteredDeepseekV41Config:
    """A registration config on one CPU device."""
    return RegisteredDeepseekV41Config(
        dtype=DType.bfloat16,
        kv_params=NonCallableMock(),
        devices=[DeviceRef.CPU()],
        max_seq_len=64,
        **fields,
    )


def byte_tokenizer() -> PreTrainedTokenizerFast:
    """A byte-level tokenizer with the placeholder as a plain added token."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(
        models.BPE(
            vocab={char: index for index, char in enumerate(alphabet)},
            merges=[],
        )
    )
    setattr(  # noqa: B010
        backend,
        "pre_tokenizer",
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    )
    setattr(backend, "decoder", decoders.ByteLevel())  # noqa: B010
    backend.add_tokens(
        [AddedToken(IMAGE_PLACEHOLDER, normalized=False, special=False)]
    )
    # The reasoning delimiters the tokenizer resolves at construction.
    backend.add_special_tokens(["<think>", "</think>"])
    return PreTrainedTokenizerFast(tokenizer_object=backend)


def png(width: int, height: int) -> bytes:
    """A black PNG of the given size."""
    data = BytesIO()
    Image.new("RGB", (width, height), color="black").save(data, format="PNG")
    return data.getvalue()


def chat_request(
    text: str, images: list[bytes] | None = None
) -> TextGenerationRequest:
    """One user message: ``text``, then one image part per image."""
    images = images or []
    message = TextGenerationRequestMessage.model_validate(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                *({"type": "image"} for _ in images),
            ],
        }
    )
    return TextGenerationRequest(
        request_id=RequestID(),
        model_name="test",
        messages=[message],
        images=images,
    )
