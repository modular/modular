# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
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
from pytest_mock import MockerFixture


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
