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
"""Unit tests for SpeculativeConfig and synthetic acceptance sampling."""

from __future__ import annotations

import pytest
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    compute_synthetic_acceptance_base_rate,
)
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


def test_synthetic_acceptance_rate_defaults_none() -> None:
    assert SpeculativeConfig().synthetic_acceptance_rate is None


def test_synthetic_acceptance_rate_valid() -> None:
    assert (
        SpeculativeConfig(
            synthetic_acceptance_rate=0.8
        ).synthetic_acceptance_rate
        == 0.8
    )
    assert (
        SpeculativeConfig(
            synthetic_acceptance_rate=0.0
        ).synthetic_acceptance_rate
        == 0.0
    )
    assert (
        SpeculativeConfig(
            synthetic_acceptance_rate=1.0
        ).synthetic_acceptance_rate
        == 1.0
    )


def test_synthetic_acceptance_rate_invalid() -> None:
    with pytest.raises(Exception):
        SpeculativeConfig(synthetic_acceptance_rate=-0.1)
    with pytest.raises(Exception):
        SpeculativeConfig(synthetic_acceptance_rate=1.5)


@pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 7, 10])
@pytest.mark.parametrize("rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99])
def test_compute_synthetic_acceptance_base_rate(
    num_steps: int, rate: float
) -> None:
    """Verify calibration produces a base_rate matching the target mean."""
    tol = 1e-9
    base_rate = compute_synthetic_acceptance_base_rate(rate, num_steps, tol=tol)

    mean_joint = sum(base_rate ** (i + 1) for i in range(num_steps)) / num_steps

    assert abs(rate - mean_joint) < 10 * tol
    assert 0.0 <= base_rate <= 1.0


def test_acceptance_sampler_greedy_by_default() -> None:
    sampler = AcceptanceSampler()
    assert sampler._base_rate is None


def test_acceptance_sampler_synthetic() -> None:
    """Calibration solves for base_rate so the mean joint acceptance
    matches the target rate."""
    rate = 0.8
    num_steps = 5
    sampler = AcceptanceSampler(
        synthetic_acceptance_rate=rate, num_draft_steps=num_steps
    )
    assert sampler._base_rate is not None

    mean_joint = (
        sum(sampler._base_rate ** (i + 1) for i in range(num_steps)) / num_steps
    )
    assert abs(mean_joint - rate) < 1e-6
