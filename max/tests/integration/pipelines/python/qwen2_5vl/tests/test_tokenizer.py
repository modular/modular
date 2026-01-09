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

"""Tests for Qwen2.5VL tokenizer."""

from unittest.mock import MagicMock

import pytest
from max.interfaces import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
)
from max.pipelines.architectures.qwen2_5vl.tokenizer import Qwen2_5VLTokenizer
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from pytest_mock import MockerFixture
from transformers import AutoConfig


class MockVisionConfig(AutoConfig):
    def __init__(self):
        self.tokens_per_second = 50


class MockKVCacheConfig(KVCacheConfig):
    def __init__(self):
        self.enable_prefix_caching = True


class MockHuggingFaceConfig(AutoConfig):
    def __init__(self):
        self.image_token_id = 128253
        self.video_token_id = 128254
        self.vision_start_token_id = 128255
        self.vision_config = MockVisionConfig()


class MockModelConfig(MAXModelConfig):
    def __init__(self):
        self._kv_cache = MockKVCacheConfig()
        self._huggingface_config = MockHuggingFaceConfig()


class MockPipelineConfig(PipelineConfig):
    def __init__(self):
        self._model_config = MockModelConfig()


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_initialization() -> None:
    """Test tokenizer initialization."""

    pipeline_config = MockPipelineConfig()
    tokenizer = Qwen2_5VLTokenizer(
        "HuggingFaceM4/Idefics3-8B-Llama3", pipeline_config=pipeline_config
    )
    assert tokenizer.image_token_id == 128253
    assert tokenizer.video_token_id == 128254
    assert tokenizer.vision_start_token_id == 128255
    assert tokenizer.enable_prefix_caching is True
    assert tokenizer.tokens_per_second == 50


@pytest.mark.asyncio
async def test_qwen2_5vl_tokenizer_new_context_smoke(
    mocker: MockerFixture,
) -> None:
    """Smoke test to ensure new_context() doesn't raise"""

    mock_tok = MagicMock()
    mock_tok.model_max_length = 2048
    mock_tok.eos_token_id = 2
    mock_tok.apply_chat_template.return_value = "test prompt"
    mock_tok.return_value = {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
    }
    mocker.patch(
        "max.pipelines.architectures.qwen2_5vl.tokenizer.AutoTokenizer.from_pretrained",
        return_value=mock_tok,
    )

    cfg = MagicMock()
    cfg.vision_config = MagicMock()
    mocker.patch(
        "max.pipelines.architectures.qwen2_5vl.tokenizer.AutoConfig.from_pretrained",
        return_value=cfg,
    )

    tokenizer = Qwen2_5VLTokenizer("test-model")

    # Attributes normally injected via PipelineConfig; set minimally for the smoke test.
    tokenizer.image_token_id = 99999
    tokenizer.video_token_id = -1
    tokenizer.vision_start_token_id = -2
    tokenizer.tokens_per_second = 50

    request = TextGenerationRequest(
        messages=[TextGenerationRequestMessage(role="user", content="test")],
        request_id=RequestID("test-id"),
        model_name="test-model",
    )

    context = await tokenizer.new_context(request)
    assert context is not None
