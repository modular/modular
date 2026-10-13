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

"""The token-shard count the graph uses must be the one EP sizes buffers from.

GLM-5.3-Flash carries its residual as one token shard per rank under
tensor-parallel attention, so the MoE dispatches a shard rather than every
rank's copy of the whole sequence. The EP dispatch buffers are sized by
:func:`calculate_ep_max_tokens_per_rank`, which divides by ``ep_size //
data_parallel_degree``. If those two ever disagree the model allocates
dispatch buffers for a token count nobody reserved, which is what the
architecture-specific override this replaced was working around.
"""

from __future__ import annotations

import pytest
from max.nn.comm.ep.ep_config import calculate_ep_max_tokens_per_rank
from max.pipelines.architectures.glm5_next.model_config import (
    token_shard_degree,
)
from max.support.math import ceildiv


@pytest.mark.parametrize("num_devices", [1, 2, 4, 8])
def test_shards_match_ep_sizing_under_tensor_parallel_attention(
    num_devices: int,
) -> None:
    """``ep_size == num_devices`` and DP 1 is the deployed TP_EP shape."""
    degree = token_shard_degree(num_devices, data_parallel_degree=1)
    assert degree == num_devices

    max_batch_input_tokens = 8192
    assert ceildiv(max_batch_input_tokens, degree) == (
        calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=max_batch_input_tokens,
            ep_size=num_devices,
            data_parallel_degree=1,
        )
    )


@pytest.mark.parametrize("num_devices", [1, 2, 4, 8])
def test_data_parallel_attention_does_not_shard_tokens(
    num_devices: int,
) -> None:
    """Each rank already owns a distinct batch shard, so the axis stays put."""
    assert token_shard_degree(num_devices, num_devices) == 1

    max_batch_input_tokens = 8192
    assert max_batch_input_tokens == calculate_ep_max_tokens_per_rank(
        max_batch_input_tokens=max_batch_input_tokens,
        ep_size=num_devices,
        data_parallel_degree=num_devices,
    )


def test_single_device_never_shards() -> None:
    assert token_shard_degree(1, data_parallel_degree=1) == 1


@pytest.mark.parametrize(
    ("total_tokens", "degree"),
    [(4, 8), (8, 8), (9, 8), (1, 2), (8192, 8)],
)
def test_shard_bound_covers_the_largest_bin(
    total_tokens: int, degree: int
) -> None:
    """``ops.reducescatter.sum`` gives rank ``r`` ``ceil((T - r) / G)`` rows.

    The EP cap is a single number for every rank, so it has to be rank 0's
    bin. A batch smaller than the device count leaves the tail ranks with
    zero rows, which a small decode batch reaches routinely.
    """
    bins = [(total_tokens + (degree - r - 1)) // degree for r in range(degree)]
    assert sum(bins) == total_tokens
    assert max(bins) == ceildiv(total_tokens, degree)
    assert bins[0] == max(bins)
    assert min(bins) >= 0
