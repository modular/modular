# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from max.serve.scheduler import TokenGenerationSchedulerConfig


def test_scheduler_max_batch_size_less_than_target_tokens_per_batch_ce() -> (
    None
):
    # ok
    TokenGenerationSchedulerConfig(
        max_batch_size=100,
        target_tokens_per_batch_ce=100,
        max_forward_steps_tg=10,
    )

    # not ok because max_batch_size > target_tokens_per_batch_ce
    with pytest.raises(
        ValueError,
        match=r"`max_batch_size` must be less than or equal to `target_tokens_per_batch_ce`, found 101 > 100",
    ):
        TokenGenerationSchedulerConfig(
            max_batch_size=101,
            target_tokens_per_batch_ce=100,
            max_forward_steps_tg=10,
        )
