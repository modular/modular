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
"""Tests for rejection_sampler (speculative decoding)."""

import numpy as np
import pytest
from max.driver import Buffer
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.lib import rejection_sampler


@pytest.fixture(scope="module")
def compiled_rejection_sampler(session: InferenceSession) -> Model:
    """Compile rejection_sampler once for the module."""
    device = session.devices[0]
    graph = rejection_sampler(device=DeviceRef.from_device(device))
    return session.load(graph)


def test_rejection_sampler(
    session: InferenceSession, compiled_rejection_sampler: Model
) -> None:
    device = session.devices[0]

    # Variables
    batch_size = 3
    num_steps = 5
    vocab_size = 10

    # Generate Random Logits and Pass Through
    draft_logits = np.random.default_rng().random(
        size=(batch_size, num_steps), dtype=np.float32
    )
    draft_tokens = np.random.randint(
        0, vocab_size, size=(batch_size, num_steps)
    )
    target_logits = np.random.default_rng().random(
        size=(batch_size * (num_steps + 1), vocab_size),
        dtype=np.float32,
    )
    target_logit_offsets = np.arange(
        0, (batch_size + 1) * (num_steps + 1), num_steps + 1
    )

    first_rejected_token, sampled_tokens = compiled_rejection_sampler(
        Buffer.from_dlpack(draft_tokens).to(device),
        Buffer.from_dlpack(draft_logits).to(device),
        Buffer.from_dlpack(target_logits).to(device),
        Buffer.from_dlpack(target_logit_offsets).to(device),
        Buffer.from_numpy(np.array([0], dtype=np.uint64)).to(device),
    )
    assert isinstance(first_rejected_token, Buffer)
    assert isinstance(sampled_tokens, Buffer)

    # Bring these back to CPU
    first_rejected_token_np = first_rejected_token.to_numpy()
    sampled_tokens_np = sampled_tokens.to_numpy()

    # Basic Rejection Sampler Impl in Python
    for x in range(batch_size):
        for i in range(num_steps):
            target_idx = (x * (num_steps + 1)) + i
            draft_logit = draft_logits[x][i]
            token_idx = draft_tokens[x][i]
            target_logit = target_logits[target_idx][token_idx]

            if draft_logit > target_logit:
                assert first_rejected_token_np[x][0] == i, f"x: {x}, i: {i}"
                assert (
                    np.argmax(target_logits[target_idx])
                    == sampled_tokens_np[x][0]
                ), (
                    f"target_logits: {target_logits[target_idx]}, sampled: {sampled_tokens_np[x][0]}"
                )

                break
