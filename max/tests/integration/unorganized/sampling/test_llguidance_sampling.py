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
"""Tests for llguidance-based structured output sampling."""

import json

import llguidance.hf
import llguidance.numpy
import numpy as np
import pytest
from llguidance import LLMatcher
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.interfaces import SamplingParams
from max.pipelines.lib import SamplingConfig, token_sampler
from transformers import AutoConfig, AutoTokenizer


@pytest.fixture(scope="module")
def structured_output_sampler(session: InferenceSession) -> Model:
    """Compile token_sampler with structured output enabled."""
    sampling_config = SamplingConfig(
        enable_structured_output=True,
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph = token_sampler(sampling_config, device=DeviceRef.GPU())
    return session.load(graph)


def test_llguidance_sampling(
    session: InferenceSession,
    structured_output_sampler: Model,
    modular_ai_llama_3_1_local_path: str,
) -> None:
    config = AutoConfig.from_pretrained(modular_ai_llama_3_1_local_path)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        modular_ai_llama_3_1_local_path
    )
    tokenizer = llguidance.hf.from_tokenizer(
        hf_tokenizer, n_vocab=config.vocab_size
    )

    # Compile the grammar for a sample schema.
    person_schema = {
        "title": "Person",
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {
                "type": "integer",
            },
        },
        "required": ["name", "age"],
    }

    matcher = LLMatcher(tokenizer, json.dumps(person_schema))

    device = session.devices[0]

    # Variables
    batch_size = 1
    vocab_size = tokenizer.vocab_size
    n_trials = 1

    sampling_params = SamplingParams(top_k=5)

    generated_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    temperature = Buffer.from_numpy(
        np.array([sampling_params.temperature] * batch_size, dtype=np.float32)
    ).to(device)
    top_k_np = np.array([sampling_params.top_k] * batch_size, dtype=np.int64)
    top_k = Buffer.from_numpy(top_k_np).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k_np), dtype=np.int64))
    top_p = Buffer.from_numpy(
        np.array([sampling_params.top_p] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(
        np.array(sampling_params.top_p, dtype=np.float32)
    )
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)
    seed = Buffer.from_numpy(
        np.array([sampling_params.seed] * batch_size, dtype=np.uint64)
    ).to(device)
    for _ in range(n_trials):
        token_bitmask = llguidance.numpy.allocate_token_bitmask(
            batch_size, vocab_size
        )
        llguidance.numpy.fill_next_token_bitmask(matcher, token_bitmask)

        # Generate Random Logits
        logits = np.random.default_rng().random(
            size=(batch_size, vocab_size), dtype=np.float32
        )

        bits = 2 ** np.arange(32, dtype=np.int32)
        bitmask = (np.expand_dims(token_bitmask, axis=-1) & bits) != 0
        bitmask = bitmask.reshape(
            batch_size,
            -1,
        ).astype(bool)

        # Run through Sampler
        _, new_tokens = structured_output_sampler(
            Buffer.from_dlpack(logits).to(device),
            generated_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p,
            seed,
            Buffer.from_dlpack(bitmask).to(device),
        )[:2]
        assert isinstance(new_tokens, Buffer)
        for token in new_tokens.to_numpy():
            assert matcher.validate_tokens(token) == len(token)
