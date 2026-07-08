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
"""Tests for the cascade ``MAXTokenizer`` worker.

The ``encode``/``decode`` paths run against a real (small) HuggingFace
tokenizer -- ``HuggingFaceTB/SmolLM2-135M-Instruct`` -- deployed on the local
runtime, and are checked against the reference ``transformers`` tokenizer.
"""

from __future__ import annotations

import hf_repo_lock
import numpy as np
import pytest
from max.experimental.cascade import LocalRuntime
from max.experimental.cascade.core.pipeline_method import _pipeline_method_scope
from max.experimental.cascade.workers.max_tokenizer import MAXTokenizer
from max.pipelines.lib import generate_local_model_path
from transformers import AutoTokenizer, PreTrainedTokenizerBase

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)


def _model_path() -> str:
    """Resolve a cached local path for the test model, else the repo id."""
    assert REVISION is not None
    try:
        return generate_local_model_path(REPO_ID, REVISION)
    except FileNotFoundError:
        # Not pre-cached; fall back to the repo id so the HF hub downloads it
        # (requires network; the bazel target is tagged ``requires-network``).
        return REPO_ID


@pytest.fixture(scope="module")
def model_path() -> str:
    return _model_path()


@pytest.mark.asyncio
async def test_open_loads_real_tokenizer(model_path: str) -> None:
    tokenizer = MAXTokenizer(model_path)
    assert tokenizer._tokenizer is None
    async with tokenizer.open():
        assert isinstance(tokenizer._tokenizer, PreTrainedTokenizerBase)


@pytest.mark.asyncio
async def test_encode_text_matches_reference(model_path: str) -> None:
    reference = AutoTokenizer.from_pretrained(model_path)
    prompt = "The quick brown fox jumps over the lazy dog."
    expected = np.asarray(reference.encode(prompt), dtype=np.int64)
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        tokens = await (await tokenizer.encode(prompt))
    assert tokens.dtype == np.int32
    assert tokens.tolist() == expected.tolist()


@pytest.mark.asyncio
async def test_encode_chat_messages_matches_reference(model_path: str) -> None:
    reference = AutoTokenizer.from_pretrained(model_path)
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
    ]
    expected = reference.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )["input_ids"]
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        tokens = await (await tokenizer.encode(messages))
    assert tokens.dtype == np.int32
    assert tokens.tolist() == list(expected)


@pytest.mark.asyncio
async def test_decode_roundtrip(model_path: str) -> None:
    reference = AutoTokenizer.from_pretrained(model_path)
    text = "Hello, cascade!"
    async with LocalRuntime() as rt, _pipeline_method_scope():
        tokenizer = await rt.deploy(MAXTokenizer(model_path))
        tokens = await (await tokenizer.encode(text))
        decoded = await (await tokenizer.decode(tokens))
    # The worker wraps the HF tokenizer, so its decode must match the reference
    # decode of the same ids.
    assert decoded == reference.decode(reference.encode(text))


@pytest.mark.asyncio
async def test_encode_before_deploy_raises() -> None:
    tokenizer = MAXTokenizer(REPO_ID)
    with pytest.raises(AssertionError, match="must be deployed"):
        await tokenizer.encode("hi")
