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
"""Checks metric reads across serving namespace generations."""

from unittest.mock import Mock, patch

import pytest
from smoke_tests.eval_runner import (
    get_num_tokens_generated,
    get_spec_decode_acceptance_totals,
)


@pytest.mark.parametrize(
    "metric",
    [
        "max_num_output_tokens_total",
        "max_request_output_tokens_total",
        "maxserve_num_output_tokens_total",
        "mach_request_output_tokens_total",
    ],
)
def test_generated_tokens(metric: str) -> None:
    response = Mock(text=f'{metric}{{model="a"}} 10\n{metric}{{model="b"}} 5\n')
    with patch("smoke_tests.eval_runner.requests.get", return_value=response):
        assert get_num_tokens_generated("http://engine/metrics") == 15


def test_acceptance_prefers_current_without_adding_aliases() -> None:
    response = Mock(
        text="max_spec_decode_avg_acceptance_length_sum 12\nmax_spec_decode_avg_acceptance_length_count 3\nmach_spec_decode_avg_acceptance_length_sum 100\nmach_spec_decode_avg_acceptance_length_count 50\n"
    )
    with patch("smoke_tests.eval_runner.requests.get", return_value=response):
        assert get_spec_decode_acceptance_totals("http://engine/metrics") == (
            12,
            3,
        )
