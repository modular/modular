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
"""Tests for top_p sampling using token_sampler."""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.lib import SamplingConfig, token_sampler


@pytest.fixture(scope="module")
def fused_sampling_model(session: InferenceSession) -> Model:
    """Compile token_sampler graph once for the module."""
    device_ref = DeviceRef.from_device(session.devices[0])
    sampling_config = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph = token_sampler(sampling_config, device=device_ref)
    return session.load(graph)


def test_top_p_sampling(
    session: InferenceSession, fused_sampling_model: Model
) -> None:
    """Test that top_p (nucleus) sampling works correctly.

    This test verifies that:
    - top_p=0.5 restricts sampling to fewer tokens
    - top_p=1.0 allows sampling from the full distribution
    - Different seeds produce different samples from the allowed tokens
    """
    device = session.devices[0]
    batch_size = 1
    vocab_size = 8
    num_trials = 100

    # Logits with a clear distribution: first two tokens have highest probability
    # After softmax with temp=1.0, roughly: [0.73, 0.27, 0.002, 0.0001, ...]
    logits_np = np.array(
        [[10.0, 9.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]],
        dtype=np.float32,
    )
    logits = Buffer.from_numpy(logits_np).to(device)

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    # top_k covers all tokens so top_p is the limiting factor
    top_k = Buffer.from_numpy(
        np.array([vocab_size] * batch_size, dtype=np.int64)
    ).to(device)
    max_k = Buffer.from_numpy(np.array(vocab_size, dtype=np.int64))
    temperature = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)

    # Test 1: top_p = 0.5 should mostly sample token 0
    top_p_restrictive = Buffer.from_numpy(
        np.array([0.5] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p_restrictive = Buffer.from_numpy(np.array(0.5, dtype=np.float32))

    sampled_tokens_restrictive: set[int] = set()
    for seed_val in range(num_trials):
        seed = Buffer.from_numpy(
            np.array([seed_val] * batch_size, dtype=np.uint64)
        ).to(device)
        tokens, _ = fused_sampling_model(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p_restrictive,
            min_top_p_restrictive,
            min_p,
            seed,
        )[:2]
        assert isinstance(tokens, Buffer)
        sampled_tokens_restrictive.add(int(tokens.to_numpy()[0]))

    # With top_p=0.5, only token 0 should be sampled (it has ~73% probability)
    assert sampled_tokens_restrictive == {0}, (
        f"top_p=0.5 should only sample the highest probability token, "
        f"got: {sampled_tokens_restrictive}"
    )

    # Test 2: top_p = 1.0 should sample from more tokens
    top_p_full = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p_full = Buffer.from_numpy(np.array(1.0, dtype=np.float32))

    sampled_tokens_full: set[int] = set()
    for seed_val in range(num_trials):
        seed = Buffer.from_numpy(
            np.array([seed_val] * batch_size, dtype=np.uint64)
        ).to(device)
        tokens, _ = fused_sampling_model(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p_full,
            min_top_p_full,
            min_p,
            seed,
        )[:2]
        assert isinstance(tokens, Buffer)
        sampled_tokens_full.add(int(tokens.to_numpy()[0]))

    # With top_p=1.0, we should see multiple tokens sampled
    assert len(sampled_tokens_full) > 1, (
        f"top_p=1.0 should sample multiple tokens, got: {sampled_tokens_full}"
    )


def test_top_p_zero_returns_argmax(
    session: InferenceSession, fused_sampling_model: Model
) -> None:
    """Test that top_p=0 returns argmax (greedy decoding)."""
    device = session.devices[0]
    batch_size = 1
    vocab_size = 8
    num_trials = 10

    # Create logits where token 3 has the highest probability
    logits_np = np.array(
        [[1.0, 2.0, 3.0, 10.0, 0.5, 0.1, -1.0, -2.0]],
        dtype=np.float32,
    )
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
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(np.array(0.0, dtype=np.float32))
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)

    for seed_val in range(num_trials):
        seed = Buffer.from_numpy(
            np.array([seed_val] * batch_size, dtype=np.uint64)
        ).to(device)
        tokens, _ = fused_sampling_model(
            logits,
            prev_tokens,
            top_k,
            max_k,
            temperature,
            top_p,
            min_top_p,
            min_p,
            seed,
        )[:2]
        assert isinstance(tokens, Buffer)
        token = int(tokens.to_numpy()[0])
        assert token == 3, f"top_p=0 should return argmax (3), got {token}"
