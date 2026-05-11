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

"""Unit tests for benchmark_shared.distribution module."""

import numpy as np
import pytest
from max.benchmark.benchmark_shared.datasets.distribution import (
    BaseDistribution,
    Burr12Distribution,
    ConstantDistribution,
    ContinuousDistribution,
    DiscreteDistribution,
    DiscreteUniformDistribution,
    GammaDistribution,
    LogNormalDistribution,
    NegativeBinomialDistribution,
    NormalDistribution,
    UniformDistribution,
)

# ---------------------------------------------------------------------------
# ConstantDistribution
# ---------------------------------------------------------------------------


def test_float_returns_constant() -> None:
    dist = BaseDistribution.from_distribution_parameter(42.0)
    assert isinstance(dist, ConstantDistribution)
    assert dist.value == 42.0


def test_str_float_returns_constant() -> None:
    dist = BaseDistribution.from_distribution_parameter("42.0")
    assert isinstance(dist, ConstantDistribution)
    assert dist.value == 42.0


def test_returns_same_value() -> None:
    dist = ConstantDistribution(value=42.0)
    assert dist.sample_value() == 42.0
    assert dist.sample_value() == 42.0


# ---------------------------------------------------------------------------
# NormalDistribution
# ---------------------------------------------------------------------------


def test_negative_std_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NormalDistribution(mean=100.0, std=-1.0)


def test_normal_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("N(100, 1)")
    assert isinstance(dist, NormalDistribution)
    assert dist.mean == 100.0
    assert dist.std == 1.0


def test_samples_near_mean_std() -> None:
    np.random.seed(0)
    dist = NormalDistribution(mean=100.0, std=1.0)
    samples = [dist.sample_value() for _ in range(500)]
    assert abs(np.mean(samples) - 100.0) < 5.0
    assert abs(np.std(samples) - 1.0) < 0.05


# ---------------------------------------------------------------------------
# UniformDistribution
# ---------------------------------------------------------------------------


def test_lower_greater_than_upper_raises() -> None:
    with pytest.raises(ValueError, match="Lower bound"):
        UniformDistribution(lower=10.0, upper=1.0)


def test_equal_bounds_returns_that_value() -> None:
    dist = UniformDistribution(lower=3.0, upper=3.0)
    assert dist.sample_value() == 3.0


def test_uniform_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("U(1, 10)")
    assert isinstance(dist, UniformDistribution)
    assert dist.lower == 1.0
    assert dist.upper == 10.0


def test_get_value_within_bounds() -> None:
    dist = UniformDistribution(lower=1.0, upper=10.0)
    for _ in range(100):
        val = dist.sample_value()
        assert 1.0 <= val <= 10.0


def test_uniform_is_continuous() -> None:
    dist = UniformDistribution(lower=0.0, upper=1.0)
    assert isinstance(dist, ContinuousDistribution)
    assert isinstance(dist, BaseDistribution)


# ---------------------------------------------------------------------------
# DiscreteUniformDistribution
# ---------------------------------------------------------------------------


def test_discrete_uniform_lower_greater_than_upper_raises() -> None:
    with pytest.raises(ValueError, match="Lower bound"):
        DiscreteUniformDistribution(lower=10, upper=1)


def test_discrete_uniform_equal_bounds() -> None:
    dist = DiscreteUniformDistribution(lower=5, upper=5)
    assert dist.sample_value() == 5


def test_discrete_uniform_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("DU(1, 10)")
    assert isinstance(dist, DiscreteUniformDistribution)
    assert dist.lower == 1
    assert dist.upper == 10


def test_discrete_uniform_returns_ints_within_bounds() -> None:
    dist = DiscreteUniformDistribution(lower=1, upper=10)
    for _ in range(200):
        val = dist.sample_value()
        assert isinstance(val, int)
        assert 1 <= val <= 10


def test_discrete_uniform_is_discrete() -> None:
    dist = DiscreteUniformDistribution(lower=0, upper=5)
    assert isinstance(dist, DiscreteDistribution)
    assert isinstance(dist, BaseDistribution)


def test_discrete_uniform_non_int_raises() -> None:
    with pytest.raises(ValueError, match="Cannot parse integer"):
        DiscreteUniformDistribution.parse_from_str_schema("DU(1.5, 3.7)")


# ---------------------------------------------------------------------------
# NegativeBinomialDistribution
# ---------------------------------------------------------------------------


def test_nb_non_positive_n_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        NegativeBinomialDistribution(n=0, p=0.5)
    with pytest.raises(ValueError, match="must be positive"):
        NegativeBinomialDistribution(n=-1, p=0.5)


def test_nb_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("NB(3, 0.4)")
    assert isinstance(dist, NegativeBinomialDistribution)
    assert dist.n == 3
    assert dist.p == 0.4


def test_nb_samples_always_at_least_one() -> None:
    dist = NegativeBinomialDistribution(n=1, p=0.99)
    for _ in range(500):
        val = dist.sample_value()
        assert isinstance(val, int)
        assert val >= 1


def test_nb_samples_near_expected_mean() -> None:
    np.random.seed(0)
    n, p = 3, 0.4
    dist = NegativeBinomialDistribution(n=n, p=p)
    # Shifted NB mean = 1 + n*(1-p)/p
    expected_mean = 1 + n * (1 - p) / p
    samples = [dist.sample_value() for _ in range(2000)]
    assert abs(np.mean(samples) - expected_mean) < 0.5


def test_nb_is_discrete() -> None:
    dist = NegativeBinomialDistribution(n=1, p=0.5)
    assert isinstance(dist, DiscreteDistribution)
    assert isinstance(dist, BaseDistribution)


# ---------------------------------------------------------------------------
# GammaDistribution
# ---------------------------------------------------------------------------


def test_non_positive_shape_raises() -> None:
    with pytest.raises(ValueError, match="Shape must be positive"):
        GammaDistribution(shape=0.0, scale=1.0)
    with pytest.raises(ValueError, match="Shape must be positive"):
        GammaDistribution(shape=-1.0, scale=1.0)


def test_non_positive_scale_raises() -> None:
    with pytest.raises(ValueError, match="Scale must be positive"):
        GammaDistribution(shape=1.0, scale=0.0)
    with pytest.raises(ValueError, match="Scale must be positive"):
        GammaDistribution(shape=1.0, scale=-1.0)


def test_gamma_string() -> None:
    dist = GammaDistribution.parse_from_str_schema("G(2, 100)")
    assert dist.shape == 2.0
    assert dist.scale == 100.0


def test_get_value_near_mean() -> None:
    dist = GammaDistribution(shape=2.0, scale=100.0)
    # Gamma(k=2, theta=100) has mean = k*theta = 200
    values = [dist.sample_value() for _ in range(500)]
    mean = float(np.mean(values))
    assert 190 < mean < 210, f"Expected mean ~200, got {mean}"


# ---------------------------------------------------------------------------
# LogNormalDistribution
# ---------------------------------------------------------------------------


def test_lognormal_negative_std_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        LogNormalDistribution(mean=0.0, std=-1.0)


def test_lognormal_string() -> None:
    dist = BaseDistribution.from_distribution_parameter("LN(1, 0.5)")
    assert isinstance(dist, LogNormalDistribution)
    assert dist.mean == 1.0
    assert dist.std == 0.5


def test_lognormal_samples_positive() -> None:
    dist = LogNormalDistribution(mean=0.0, std=1.0)
    for _ in range(100):
        assert dist.sample_value() > 0.0


def test_lognormal_samples_near_expected_mean() -> None:
    np.random.seed(0)
    dist = LogNormalDistribution(mean=1.0, std=0.5)
    # LogNormal(mu=1, sigma=0.5) has mean = exp(mu + sigma^2/2) ≈ 3.08
    expected_mean = np.exp(1.0 + 0.5**2 / 2)
    samples = [dist.sample_value() for _ in range(1000)]
    assert abs(np.mean(samples) - expected_mean) < 0.5


# ---------------------------------------------------------------------------
# BaseDistribution.from_distribution_parameter
# ---------------------------------------------------------------------------


def test_unrecognized_string_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized distribution"):
        BaseDistribution.from_distribution_parameter("1,2,3,4,5")
    with pytest.raises(ValueError, match="Unrecognized distribution"):
        BaseDistribution.from_distribution_parameter("X(1, 2)")


# ---------------------------------------------------------------------------
# Hierarchy checks
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Burr12Distribution
# ---------------------------------------------------------------------------


def test_burr12_non_positive_c_raises() -> None:
    with pytest.raises(ValueError, match="c"):
        Burr12Distribution(c=0.0, d=1.0, scale=1.0)


def test_burr12_non_positive_d_raises() -> None:
    with pytest.raises(ValueError, match="d"):
        Burr12Distribution(c=1.0, d=-0.5, scale=1.0)


def test_burr12_non_positive_scale_raises() -> None:
    with pytest.raises(ValueError, match="scale"):
        Burr12Distribution(c=1.0, d=1.0, scale=0.0)


def test_burr12_string() -> None:
    dist = BaseDistribution.from_distribution_parameter(
        "Burr12(2.389, 0.569, 214.8)"
    )
    assert isinstance(dist, Burr12Distribution)
    assert dist.c == pytest.approx(2.389)
    assert dist.d == pytest.approx(0.569)
    assert dist.scale == pytest.approx(214.8)


def test_burr12_string_lowercase() -> None:
    dist = BaseDistribution.from_distribution_parameter("burr12(2, 1, 5)")
    assert isinstance(dist, Burr12Distribution)


def test_burr12_string_missing_param_raises() -> None:
    with pytest.raises(ValueError, match="Burr12"):
        BaseDistribution.from_distribution_parameter("Burr12(1, 2)")


def test_burr12_samples_positive_and_match_median() -> None:
    np.random.seed(0)
    c, d, scale = 2.389, 0.569, 214.8
    dist = Burr12Distribution(c=c, d=d, scale=scale)
    samples = np.array([dist.sample_value() for _ in range(20_000)])
    assert (samples > 0).all()
    expected_median = scale * (2.0 ** (1.0 / d) - 1.0) ** (1.0 / c)
    assert abs(np.median(samples) - expected_median) / expected_median < 0.05


def test_continuous_distributions_are_continuous() -> None:
    for dist_cls in (
        NormalDistribution,
        UniformDistribution,
        GammaDistribution,
        LogNormalDistribution,
        Burr12Distribution,
    ):
        assert issubclass(dist_cls, ContinuousDistribution)


def test_discrete_distributions_are_discrete() -> None:
    for dist_cls in (
        DiscreteUniformDistribution,
        NegativeBinomialDistribution,
    ):
        assert issubclass(dist_cls, DiscreteDistribution)
