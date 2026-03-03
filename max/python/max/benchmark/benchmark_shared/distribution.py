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

"""Probability distribution abstractions used for parameterized values.

Supports parsing distribution specifications from strings like "N(mean, std)"
for normal distributions, "U(lower, upper)" for uniform distributions,
"G(shape, scale)" for gamma distributions, as well as plain integer values
for constant returns.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

DistributionParameter = int | str | None
"""Type alias for parameters that accept an int, a distribution string, or None."""


class BaseDistribution(ABC):
    """Abstract base class for probability distributions used in benchmarks."""

    @abstractmethod
    def sample_value(self) -> float:
        """Sample a single value from this distribution.

        Returns:
            A sampled float value (in the same unit as the distribution parameters).
        """
        ...

    @classmethod
    def from_distribution_parameter(
        cls, param: DistributionParameter
    ) -> BaseDistribution | None:
        """Parse a distribution parameter into a concrete distribution instance.

        Args:
            param: An integer (constant value), a string like "N(mean,std), "
                "U(lower,upper)", "G(shape,scale)", or None.

        Returns:
            A BaseDistribution instance, or None if param is None.

        Raises:
            ValueError: If the string format is unrecognized or unparseable.
        """
        if param is None:
            return None

        elif isinstance(param, int):
            return ConstantDistribution(value=param)

        elif isinstance(param, str):
            stripped = param.strip()

            # Try parsing as a plain integer string
            try:
                return ConstantDistribution(value=int(stripped))
            except ValueError:
                pass

            if stripped.startswith("N(") or stripped.startswith("n("):
                return NormalDistribution.parse_from_str_schema(stripped)
            elif stripped.startswith("U(") or stripped.startswith("u("):
                return UniformDistribution.parse_from_str_schema(stripped)
            elif stripped.startswith("G(") or stripped.startswith("g("):
                return GammaDistribution.parse_from_str_schema(stripped)
            else:
                raise ValueError(
                    f"Unrecognized distribution format: '{param}'. "
                    "Expected an integer, 'N(mean,std)', "
                    "'U(lower,upper)', or 'G(shape,scale)'."
                )
        else:
            raise TypeError(
                "Expected int, str, or None for distribution parameter, got "
                "{type(param)}"
            )


@dataclass
class ConstantDistribution(BaseDistribution):
    """A degenerate distribution that always returns the same value."""

    value: float

    def sample_value(self) -> float:
        return self.value


@dataclass
class NormalDistribution(BaseDistribution):
    """A normal (Gaussian) distribution parameterized by mean and std."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if self.std < 0:
            raise ValueError(
                f"Standard deviation must be non-negative, got {self.std}"
            )

    def sample_value(self) -> float:
        return float(np.random.normal(loc=self.mean, scale=self.std))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> NormalDistribution:
        """Parse a string like "N(mean, std)" into a NormalDistribution.

        Args:
            schema: A string in the format "N(mean, std)" (case-insensitive prefix).

        Returns:
            A NormalDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        match = re.match(r"[Nn]\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", schema)
        if not match:
            raise ValueError(
                f"Cannot parse normal distribution from '{schema}'. "
                "Expected format: 'N(mean, std)'."
            )
        try:
            mean = float(match.group(1))
            std = float(match.group(2))
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(mean=mean, std=std)


@dataclass
class UniformDistribution(BaseDistribution):
    """A uniform distribution parameterized by lower and upper bounds."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError(
                f"Lower bound ({self.lower}) must be <= upper bound"
                f" ({self.upper})"
            )

    def sample_value(self) -> float:
        return float(np.random.uniform(low=self.lower, high=self.upper))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> UniformDistribution:
        """Parse a string like "U(lower, upper)" into a UniformDistribution.

        Args:
            schema: A string in the format "U(lower, upper)" (case-insensitive prefix).

        Returns:
            A UniformDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        match = re.match(r"[Uu]\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", schema)
        if not match:
            raise ValueError(
                f"Cannot parse uniform distribution from '{schema}'. "
                "Expected format: 'U(lower, upper)'."
            )
        try:
            lower = float(match.group(1))
            upper = float(match.group(2))
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(lower=lower, upper=upper)


@dataclass
class GammaDistribution(BaseDistribution):
    """A gamma distribution parameterized by shape (k) and scale (theta)."""

    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0:
            raise ValueError(f"Shape must be positive, got {self.shape}")
        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

    def sample_value(self) -> float:
        return float(np.random.gamma(shape=self.shape, scale=self.scale))

    @classmethod
    def parse_from_str_schema(cls, schema: str) -> GammaDistribution:
        """Parse a string like "G(shape, scale)" into a GammaDistribution.

        Args:
            schema: A string in the format "G(shape, scale)" (case-insensitive prefix).

        Returns:
            A GammaDistribution instance.

        Raises:
            ValueError: If the string cannot be parsed.
        """
        match = re.match(r"[Gg]\(\s*([^,]+)\s*,\s*([^)]+)\s*\)", schema)
        if not match:
            raise ValueError(
                f"Cannot parse gamma distribution from '{schema}'. "
                "Expected format: 'G(shape, scale)'."
            )
        try:
            shape = float(match.group(1))
            scale = float(match.group(2))
        except ValueError as e:
            raise ValueError(
                f"Cannot parse numeric values from '{schema}': {e}"
            ) from e
        return cls(shape=shape, scale=scale)
