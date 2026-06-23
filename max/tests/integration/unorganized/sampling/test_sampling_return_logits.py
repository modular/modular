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
"""Tests for token_sampler with return_all_top_k_logits option."""

import numpy as np
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.pipelines.lib import SamplingConfig, token_sampler


@pytest.fixture(scope="module")
def return_logits_sampler(session: InferenceSession) -> Model:
    """Compile token_sampler for the module.

    Note: The return_all_top_k_logits and max_top_k parameters are not yet
    available in SamplingConfig. This fixture uses a basic config for now.
    The test is skipped anyway per MXSERV-41.
    """
    device_ref = DeviceRef.from_device(session.devices[0])
    sampling_config = SamplingConfig(
        in_dtype=DType.float32,
        out_dtype=DType.float32,
    )
    graph = token_sampler(sampling_config, device=device_ref)
    return session.load(graph)


@pytest.mark.skip(
    reason=(
        "MXSERV-41: Test skipped because token_sampler with "
        "return_all_top_k_logits=True returns incorrect logits shape"
    )
)
def test_sampling_return_logits(
    session: InferenceSession, return_logits_sampler: Model
) -> None:
    """Test that token_sampler returns top-k logits when requested.

    When return_all_top_k_logits=True, the sampler should return:
    - The sampled tokens
    - Updated previous tokens
    - The top-k logits for each position
    """
    device = session.devices[0]
    batch_size = 2
    vocab_size = 16
    max_top_k = 8

    # Create random logits
    np.random.seed(42)
    logits_np = np.random.randn(batch_size, vocab_size).astype(np.float32)
    logits = Buffer.from_numpy(logits_np).to(device)

    prev_tokens = Buffer(
        shape=(batch_size, 0),
        dtype=DType.int64,
        device=device,
    )

    top_k = np.array([max_top_k] * batch_size, dtype=np.int64)
    top_k_tensor = Buffer.from_numpy(top_k).to(device)
    max_k = Buffer.from_numpy(np.array(np.max(top_k), dtype=np.int64))

    temperature = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    top_p = Buffer.from_numpy(
        np.array([1.0] * batch_size, dtype=np.float32)
    ).to(device)
    min_top_p = Buffer.from_numpy(np.array(1.0, dtype=np.float32))
    min_p = Buffer.from_numpy(
        np.array([0.0] * batch_size, dtype=np.float32)
    ).to(device)
    seed = Buffer.from_numpy(np.array([42] * batch_size, dtype=np.uint64)).to(
        device
    )

    results = return_logits_sampler(
        logits,
        prev_tokens,
        top_k_tensor,
        max_k,
        temperature,
        top_p,
        min_top_p,
        min_p,
        seed,
    )

    # Should return: tokens, updated_prev_tokens, top_k_logits
    assert len(results) >= 3, f"Expected at least 3 outputs, got {len(results)}"

    tokens, updated_prev_tokens, top_k_logits = results[:3]

    assert isinstance(tokens, Buffer)
    assert isinstance(updated_prev_tokens, Buffer)
    assert isinstance(top_k_logits, Buffer)

    tokens_np = tokens.to_numpy()
    top_k_logits_np = top_k_logits.to_numpy()

    assert tokens_np.shape == (batch_size,)
    assert top_k_logits_np.shape == (batch_size, max_top_k), (
        f"Expected top_k_logits shape ({batch_size}, {max_top_k}), "
        f"got {top_k_logits_np.shape}"
    )

    # Verify that the returned top-k logits are the actual top-k values
    for b in range(batch_size):
        expected_top_k = np.sort(logits_np[b])[-max_top_k:][::-1]
        actual_top_k = np.sort(top_k_logits_np[b])[::-1]
        np.testing.assert_allclose(
            actual_top_k,
            expected_top_k,
            rtol=1e-4,
            err_msg=f"Top-k logits mismatch for batch {b}",
        )
