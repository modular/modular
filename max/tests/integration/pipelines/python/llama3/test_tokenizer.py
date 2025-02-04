# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import text
from max.pipelines.dataprocessing.tokenizer import tokenizer_from_gguf
from transformers import AutoTokenizer


@pytest.fixture
def hf_tokenizer(testdata_directory):
    return AutoTokenizer.from_pretrained(testdata_directory)


@pytest.fixture
def tokenizer(testdata_directory):
    gguf_path = testdata_directory / "tiny_llama.gguf"
    return tokenizer_from_gguf(gguf_path)


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(s=text())
def test_tokenizer(hf_tokenizer, tokenizer, s):
    expected = hf_tokenizer.encode(s)
    actual = tokenizer.encode(s)
    assert actual == expected
