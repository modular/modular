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
"""The shared contracts the image path relies on.

Every test here is CPU-only and compiles nothing: module registration's
decoder and ``VISION`` role inputs, the vision cache's encode protocol, and
the ``--tokenizer-impl`` chat encoder of the text tokenizer.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import abi_contract
import numpy as np
import pytest
from max import functional as F
from max.driver import CPU, Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    IMAGE_PLACEHOLDER,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.registration import (
    DeepseekV41VisionEncoder,
    RegisteredDeepseekV41BatchProcessor,
    RegisteredDeepseekV41Decoder,
)
from max.pipelines.architectures.deepseekV41_modulev3.tokenizer import (
    DeepseekV41Tokenizer,
    encode_literal_text,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib import KVCacheConfig
from max.pipelines.lib.module_registration import (
    decoder_input_types,
    decoder_vision_input_types,
)
from max.tensor.sharding import DeviceMesh
from transformers import PretrainedConfig
from vision_fixtures import (
    byte_tokenizer,
    chat_request,
    png,
    registered_config,
    vision_config,
)


def test_decoder_takes_the_image_rows_after_every_text_input() -> None:
    """The text inputs keep their positions; the image rows come last."""
    assert RegisteredDeepseekV41Decoder.consumes_vision
    text = V41Config.from_hf(abi_contract.hf_config("f1"))
    kv = build_kv_params(
        text, KVCacheConfig(kv_cache_page_size=128), [DeviceRef.CPU()]
    )
    mesh = DeviceMesh((CPU(),), (1,), ("tp",))
    layout = RegisteredDeepseekV41Decoder.input_layout
    with_vision = decoder_input_types(
        kv,
        DeviceRef.CPU(),
        vision=decoder_vision_input_types(mesh, DType.bfloat16, 5120),
        input_layout=layout,
    )
    text_only = decoder_input_types(kv, DeviceRef.CPU(), input_layout=layout)
    assert list(with_vision) == [*text_only, "vision"]


def test_vision_role_follows_the_deployment() -> None:
    vision = vision_config()
    assert DeepseekV41VisionEncoder.is_enabled(registered_config(vision=vision))
    assert not DeepseekV41VisionEncoder.is_enabled(registered_config())
    with F.lazy():
        encoder = DeepseekV41VisionEncoder(registered_config(vision=vision))
    assert encoder.embedding_dtype == DType.bfloat16
    assert encoder.embedding_dim == 5120
    assert len(encoder.input_types()) == 7


def test_encoding_copies_the_span_rows_to_every_device() -> None:
    """Each packed launch runs once; their rows join in launch order."""
    config = registered_config(vision=vision_config(), hidden_size=8)
    processor = RegisteredDeepseekV41BatchProcessor.__new__(
        RegisteredDeepseekV41BatchProcessor
    )
    processor.config = config
    devices: list[Device] = [CPU(), CPU()]
    empty = processor.empty_vision_embeddings(devices)
    assert [b.shape for b in empty] == [(0, 8), (0, 8)]
    assert all(b.dtype == DType.bfloat16 for b in empty)
    assert processor.vision_execute([], devices, None).embeddings is empty

    first = Buffer.from_numpy(np.full((4, 8), 1, np.float32))
    second = Buffer.from_numpy(np.full((3, 8), 2, np.float32))
    model = Mock()
    model.execute_raw.side_effect = [[first], [second]]
    processor.bind_vision_encoder(vision_model=model, session=Mock())
    packed = [
        [Buffer.zeros([1], DType.int64)],
        [Buffer.zeros([2], DType.int64)],
    ]
    result = processor.vision_execute([], devices, packed)
    assert [call.args for call in model.execute_raw.call_args_list] == [
        tuple(launch) for launch in packed
    ]
    assert [b.shape for b in result.embeddings] == [(7, 8), (7, 8)]
    np.testing.assert_array_equal(
        result.embeddings[1].to_numpy()[:, 0], [1, 1, 1, 1, 2, 2, 2]
    )


class _StubChatEncoder:
    """A ``--tokenizer-impl`` encoder that encodes every prompt as one id."""

    digest: ClassVar[str] = ""
    prompts: ClassVar[list[str]] = []

    def tokenizer_json_sha256(self) -> str:
        return self.digest

    def encode(self, text: str) -> bytes:
        self.prompts.append(text)
        return np.array([7], dtype="<u4").tobytes()


def _served_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DeepseekV41Tokenizer:
    """A tokenizer that serves images, with the stub as ``--tokenizer-impl``."""
    byte_tokenizer().save_pretrained(tmp_path)
    monkeypatch.setattr(
        _StubChatEncoder,
        "digest",
        hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(_StubChatEncoder, "prompts", [])
    full = abi_contract.hf_config("full")
    pipeline = Mock()
    pipeline.tokenizer_impl = f"{__name__}:{_StubChatEncoder.__name__}"
    pipeline.draft_model = None
    pipeline.speculative = None
    pipeline.runtime.vision_cache_utilization = 0.05
    pipeline.model.vision_config_overrides = {}
    pipeline.model.huggingface_config = PretrainedConfig(
        text_config=PretrainedConfig(hidden_size=8),
        vision_config=PretrainedConfig(**full["vision_config"]),
        image_token_id=byte_tokenizer().convert_tokens_to_ids(
            IMAGE_PLACEHOLDER
        ),
        eos_token_id=0,
    )
    pipeline.model.generation_config = SimpleNamespace(eos_token_id=None)
    return DeepseekV41Tokenizer(str(tmp_path), pipeline, max_length=4096)


def test_the_chat_encoder_encodes_text_chats_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--tokenizer-impl`` encodes the chats with no image.

    A chat with an image, or with the placeholder in its text, encodes with
    HuggingFace, which keeps a placeholder in caller text as text.
    """
    tokenizer = _served_tokenizer(tmp_path, monkeypatch)
    assert tokenizer.expects_content_wrapping
    assert tokenizer.__getstate__()["_chat_encoder"] is None

    text = asyncio.run(tokenizer.new_context(chat_request("hello")))
    assert text.tokens.array.tolist() == [7]
    assert len(_StubChatEncoder.prompts) == 1
    assert tokenizer.take_chat_encoder_outcomes() == {"custom": 1}

    request = chat_request(f"a {IMAGE_PLACEHOLDER} b")
    literal = asyncio.run(tokenizer.new_context(request))
    assert literal.tokens.array.tolist() == encode_literal_text(
        tokenizer.delegate,
        tokenizer.apply_chat_template(request.messages),
        add_special_tokens=False,
    )
    assert tokenizer.image_token_id not in literal.tokens.array

    image = asyncio.run(
        tokenizer.new_context(chat_request("hello", [png(42, 42)]))
    )
    assert tokenizer.image_token_id in image.tokens.array
    assert len(_StubChatEncoder.prompts) == 1
    assert tokenizer.take_chat_encoder_outcomes() == {}


def test_token_prompt_longer_than_the_context_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = _served_tokenizer(tmp_path, monkeypatch)
    tokenizer.max_length = 2
    assert asyncio.run(tokenizer.encode([1, 2])).tolist() == [1, 2]
    with pytest.raises(PromptTooLongError):
        asyncio.run(tokenizer.encode([1, 2, 3]))
