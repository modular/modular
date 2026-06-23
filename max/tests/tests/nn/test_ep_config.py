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

import pytest
from max.nn.comm.ep import calculate_ep_max_tokens_per_rank


@pytest.mark.parametrize(
    "max_batch_input_tokens, ep_size, data_parallel_degree, expected",
    [
        # Evenly divisible: ceil == floor.
        (4096, 8, 1, 512),
        (1024, 4, 1, 256),
        # Non-divisible: must ceil to match ops.reducescatter.sum's
        # ceiling-biased ragged binning, which puts ceil(S/P) tokens on the
        # first (S % P) ranks. Floor would under-size the EP per-rank cap
        # and trip the dispatch assertion in ep.mojo for the over-sized
        # shards.
        (4196, 8, 1, 525),
        (4097, 8, 1, 513),
        (10, 3, 1, 4),
        # DP_EP: tp_size == 1, every rank holds the full batch.
        (4196, 8, 8, 4196),
    ],
)
def test_calculate_ep_max_tokens_per_rank_ceil(
    max_batch_input_tokens: int,
    ep_size: int,
    data_parallel_degree: int,
    expected: int,
) -> None:
    assert (
        calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=max_batch_input_tokens,
            ep_size=ep_size,
            data_parallel_degree=data_parallel_degree,
        )
        == expected
    )


def test_calculate_ep_max_tokens_per_rank_allreduce_bypasses_tp() -> None:
    # use_allreduce keeps the full batch on every rank, regardless of tp_size.
    assert (
        calculate_ep_max_tokens_per_rank(
            max_batch_input_tokens=4196,
            ep_size=8,
            data_parallel_degree=1,
            use_allreduce=True,
        )
        == 4196
    )
