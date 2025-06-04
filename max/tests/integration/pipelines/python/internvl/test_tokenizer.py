# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Minimal smoke test for InternVL tokenizer."""

from unittest.mock import MagicMock, patch

import pytest
from max.pipelines import TokenGeneratorRequest, TokenGeneratorRequestMessage
from max.pipelines.architectures.internvl.tokenizer import InternVLTokenizer


@pytest.mark.asyncio
async def test_internvl_tokenizer_new_context_smoke():
    """Smoke test to ensure new_context() doesn't raise"""
    with (
        patch(
            "max.pipelines.architectures.internvl.tokenizer.AutoTokenizer"
        ) as mock_auto_tokenizer,
        patch(
            "max.pipelines.architectures.internvl.tokenizer.AutoConfig"
        ) as mock_auto_config,
    ):
        # Create minimal mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.model_max_length = 2048
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.apply_chat_template.return_value = "test prompt"

        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_auto_config.from_pretrained.return_value = MagicMock()

        tokenizer = InternVLTokenizer("test-model")

        # Mock the processor to return expected format
        tokenizer.processor = MagicMock()
        tokenizer.processor.return_value = {
            "input_ids": [1, 2, 3],
        }

        request = TokenGeneratorRequest(
            messages=[
                TokenGeneratorRequestMessage(role="user", content="test")
            ],
            index=0,
            id="test-id",
            model_name="test-model",
        )

        context = await tokenizer.new_context(request)

        assert context is not None
