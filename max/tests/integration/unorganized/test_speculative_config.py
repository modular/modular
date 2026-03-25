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
"""Unit tests for SpeculativeConfig."""

from __future__ import annotations

from max.pipelines.lib.config.speculative_config import SpeculativeConfig


def test_is_eagle() -> None:
    """Verify is_eagle() returns correct boolean."""
    assert SpeculativeConfig(speculative_method="eagle").is_eagle()
    assert not SpeculativeConfig(speculative_method="standalone").is_eagle()
    assert not SpeculativeConfig(speculative_method=None).is_eagle()


def test_is_standalone() -> None:
    """Verify is_standalone() returns correct boolean."""
    assert SpeculativeConfig(speculative_method="standalone").is_standalone()
    assert not SpeculativeConfig(speculative_method="eagle").is_standalone()
    assert not SpeculativeConfig(speculative_method=None).is_standalone()


def test_num_speculative_tokens() -> None:
    """Verify num_speculative_tokens uses default and accepts custom values."""
    assert SpeculativeConfig().num_speculative_tokens == 2
    assert (
        SpeculativeConfig(num_speculative_tokens=10).num_speculative_tokens
        == 10
    )
    assert (
        SpeculativeConfig(num_speculative_tokens=1).num_speculative_tokens == 1
    )


def test_rejection_sampling_strategy_default() -> None:
    """Verify rejection_sampling_strategy defaults to None (resolved later based on method)."""
    config = SpeculativeConfig()
    assert config.rejection_sampling_strategy is None


def test_rejection_sampling_strategy_values() -> None:
    """Verify rejection_sampling_strategy accepts valid values."""
    assert (
        SpeculativeConfig(
            rejection_sampling_strategy="greedy"
        ).rejection_sampling_strategy
        == "greedy"
    )
    assert (
        SpeculativeConfig(
            rejection_sampling_strategy="residual"
        ).rejection_sampling_strategy
        == "residual"
    )


def test_uses_greedy_rejection() -> None:
    """Verify uses_greedy_rejection() returns correct boolean."""
    assert SpeculativeConfig(
        rejection_sampling_strategy="greedy"
    ).uses_greedy_rejection()
    assert not SpeculativeConfig(
        rejection_sampling_strategy="residual"
    ).uses_greedy_rejection()
    assert not SpeculativeConfig().uses_greedy_rejection()
