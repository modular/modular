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
"""Tests for SamplingParams validation (no GPU required)."""

import pytest
from max.interfaces import SamplingParams


def test_sampling_top_k() -> None:
    """Test that SamplingParams accepts large top_k values.

    The arbitrary limit of 255 for top_k has been removed to support models
    like Kimi 2.5 that may request larger top_k values. The underlying Mojo
    kernel's ternary search algorithm supports arbitrary k values.
    """
    # Large top_k values should be accepted (previously raised ValueError)
    params = SamplingParams(top_k=257)
    assert params.top_k == 257

    # Very large top_k values should also be accepted
    params = SamplingParams(top_k=1000)
    assert params.top_k == 1000

    # top_k=0 should be converted to -1 (sample all tokens)
    params = SamplingParams(top_k=0)
    assert params.top_k == -1

    # top_k=-1 (sample all tokens) should still work
    params = SamplingParams(top_k=-1)
    assert params.top_k == -1

    # top_k < -1 should still be rejected
    with pytest.raises(ValueError, match="top_k must be -1 or greater than 0"):
        SamplingParams(top_k=-2)


def test_temperature_zero_sets_top_k_to_one() -> None:
    sampling_params = SamplingParams(temperature=0.0)
    assert sampling_params.top_k == 1
