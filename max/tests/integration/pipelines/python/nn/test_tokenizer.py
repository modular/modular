# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from max.pipelines import (
    PipelineConfig,
    SupportedEncoding,
    TextContext,
    TextTokenizer,
)


@pytest.mark.asyncio
async def test_tokenizer__encode_and_decode():
    encoding = SupportedEncoding.q4_k
    tokenizer = TextTokenizer(
        PipelineConfig(
            architecture="llama",
            version="3.1",
            quantization_encoding=encoding,
            huggingface_repo_id="modularai/llama-3.1",
        )
    )

    test_string = "hi my name is"
    context = TextContext(max_tokens=10, cache_seq_id=0, prompt=test_string)
    encoded = await tokenizer.encode(test_string)
    decoded = await tokenizer.decode(context, encoded, skip_special_tokens=True)
    assert test_string == decoded
