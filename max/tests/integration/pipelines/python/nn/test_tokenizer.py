# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max.pipelines import SupportedEncoding, TextTokenizer
from max.pipelines.core import TextContext


@pytest.mark.asyncio
async def test_tokenizer__encode_and_decode():
    encoding = SupportedEncoding.q4_k
    tokenizer = TextTokenizer(model_path="modularai/llama-3.1")

    test_string = "hi my name is"
    encoded = await tokenizer.encode(test_string, add_special_tokens=False)
    context = TextContext(
        max_length=10,
        cache_seq_id=0,
        prompt=test_string,
        tokens=np.array(encoded),
    )
    assert context.current_length == len(encoded)
    decoded = await tokenizer.decode(context, encoded)
    assert test_string == decoded
