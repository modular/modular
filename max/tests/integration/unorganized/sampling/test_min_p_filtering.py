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
"""Tests for min_p filtering using token_sampler."""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.lib import SamplingConfig, token_sampler


@pytest.fixture(scope="module")
def min_p_filtering_model(session: InferenceSession) -> Model:
    """Compile token_sampler graph with min_p support once for the module."""
    device_ref = DeviceRef.from_device(session.devices[0])
    sampling_config = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph = token_sampler(sampling_config, device=device_ref)
    return session.load(graph)


@pytest.mark.parametrize("vocab_size", [8, 128])
def test_min_p_filtering(
    session: InferenceSession, min_p_filtering_model: Model, vocab_size: int
) -> None:
    """Test that min_p filtering works correctly.

    min_p filters out tokens whose probability is less than min_p * max_prob.
    For example, if the max probability is 0.73 and min_p=0.1, tokens with
    probability < 0.073 will be filtered out.
    """
    device = session.devices[0]
    batch_size = 1
    num_trials = 100

    # Create logits that produce a clear probability distribution
    # After softmax: roughly [0.73, 0.27, 0.002, 0.0007, 0.0003, ...]
    # Pad with very negative values if vocab_size > 8
    base_logits = [10.0, 9.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]
    if vocab_size > 8:
        base_logits.extend([-10.0] * (vocab_size - 8))
    logits_np = np.array([base_logits], dtype=np.float32)
    logits = Buffer.from_numpy(logits_np).to(device)

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    top_k = Buffer.from_numpy(
        np.array([vocab_size] * batch_size, dtype=np.int64)
    ).to(device)
    max_k = Buffer.from_numpy(np.array(vocab_size, dtype=np.int64))
    temperature = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    top_p = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(np.array(1.0, dtype=np.float32))

    # Test 1: min_p = 0.3 should only allow tokens 0 and 1
    # max_prob ~= 0.73, threshold = 0.73 * 0.3 ~= 0.22
    # Only token 0 (0.73) and token 1 (0.27) pass the threshold
    min_p_restrictive = Buffer.from_numpy(
        np.array([0.3] * batch_size, dtype=np.float32)
    ).to(device)

    sampled_tokens_restrictive: set[int] = set()
    for seed_val in range(num_trials):
        seed = Buffer.from_numpy(
            np.array([seed_val] * batch_size, dtype=np.uint64)
        ).to(device)
        tokens, _ = min_p_filtering_model(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p_restrictive,
            seed,
        )[:2]
        assert isinstance(tokens, Buffer)
        sampled_tokens_restrictive.add(int(tokens.to_numpy()[0]))

    assert sampled_tokens_restrictive == {0, 1}, (
        f"min_p=0.3 should only sample tokens 0 and 1, "
        f"got: {sampled_tokens_restrictive}"
    )

    # Test 2: min_p = 0.0 should allow all tokens (no filtering)
    min_p_none = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)

    sampled_tokens_none: set[int] = set()
    for seed_val in range(num_trials):
        seed = Buffer.from_numpy(
            np.array([seed_val] * batch_size, dtype=np.uint64)
        ).to(device)
        tokens, _ = min_p_filtering_model(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p_none,
            seed,
        )[:2]
        assert isinstance(tokens, Buffer)
        sampled_tokens_none.add(int(tokens.to_numpy()[0]))

    # With min_p=0.0, we should see at least the top 2 tokens sampled
    assert len(sampled_tokens_none) >= 2, (
        f"min_p=0.0 should sample multiple tokens, got: {sampled_tokens_none}"
    )

    # Test 3: min_p = 1.0 should only allow the max probability token
    min_p_max = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)

    for seed_val in range(num_trials):
        seed = Buffer.from_numpy(
            np.array([seed_val] * batch_size, dtype=np.uint64)
        ).to(device)
        tokens, _ = min_p_filtering_model(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p_max,
            seed,
        )[:2]
        assert isinstance(tokens, Buffer)
        token = int(tokens.to_numpy()[0])
        assert token == 0, f"min_p=1.0 should only sample token 0, got {token}"
