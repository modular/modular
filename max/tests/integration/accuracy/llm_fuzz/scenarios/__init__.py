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
"""Pluggable scenario architecture for LLM endpoint fuzz testing.

To create a new scenario:
1. Create a new .py file in this directory (or a subdirectory)
2. Subclass `BaseScenario`
3. Decorate with `@register_scenario`

The scenario will be automatically discovered and available in the CLI.
"""

from __future__ import annotations

import abc
import importlib
import pathlib
import time
from collections.abc import Awaitable, Coroutine
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from client import FuzzClient, RunConfig

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCENARIO_REGISTRY: dict[str, type[BaseScenario]] = {}
_DISCOVERED = False


def register_scenario(cls: type[BaseScenario]) -> type[BaseScenario]:
    """Decorator that registers a scenario class by its `name` attribute."""
    _SCENARIO_REGISTRY[cls.name] = cls
    return cls


def _ensure_discovered() -> None:
    """Idempotently load every scenario submodule, populating the registry.

    Discovery is deferred (rather than running at package import time)
    so that ``reporting`` — which imports ``Verdict``/``ScenarioResult``
    from this module — can finish initializing before any submodule
    reaches back into it. Eager discovery created a load-order cycle
    (reporting -> scenarios.__init__ -> submodule -> reporting) that
    fired whenever ``reporting`` was imported first.
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    pkg_dir = pathlib.Path(__file__).parent

    # Import top-level scenario modules.
    for f in sorted(pkg_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        importlib.import_module(f".{f.stem}", package=__name__)

    # Import scenario modules from subdirectories (validation/, models/, etc.).
    for subdir in sorted(pkg_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        # Skip directories without __init__.py (not a Python package).
        if not (subdir / "__init__.py").exists():
            continue
        sub_package = f"{__name__}.{subdir.name}"
        importlib.import_module(sub_package)
        for f in sorted(subdir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            importlib.import_module(f".{f.stem}", package=sub_package)


def get_all_scenarios() -> dict[str, type[BaseScenario]]:
    _ensure_discovered()
    return dict(_SCENARIO_REGISTRY)


def get_scenario(name: str) -> type[BaseScenario]:
    _ensure_discovered()
    return _SCENARIO_REGISTRY[name]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    PASS = (
        "PASS"  # Server handled it gracefully (proper error or valid response)
    )
    FAIL = "FAIL"  # Server crashed, hung, or returned corrupt data
    INTERESTING = (
        "INTERESTING"  # Server behaved unexpectedly (worth investigating)
    )
    ERROR = "ERROR"  # Our test client itself errored (network issue etc.)


@dataclass
class ScenarioResult:
    scenario_name: str
    test_name: str
    verdict: Verdict
    status_code: int | None = None
    elapsed_ms: float = 0.0
    detail: str = ""
    request_body: str = ""
    response_body: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Tracks consecutive server failures and triggers a halt."""

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self._consecutive_failures = 0
        self.tripped = False

    def record(self, result: ScenarioResult) -> None:
        if self.threshold <= 0:
            return
        if result.verdict == Verdict.FAIL:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.threshold:
                self.tripped = True
        else:
            self._consecutive_failures = 0

    def reset(self) -> None:
        self._consecutive_failures = 0
        self.tripped = False


# ---------------------------------------------------------------------------
# Base scenario
# ---------------------------------------------------------------------------


class BaseScenario(abc.ABC):
    """Base class for all fuzz-test scenarios.

    Subclasses must define:
        name: str          - unique identifier
        description: str   - what this scenario tests
        tags: list[str]    - categories for filtering

    Optional class attributes:
        requires_validator: bool  - True if scenario needs ValidatorClient (OpenAI SDK)
        model_filter: str | None  - e.g. "kimi", "glm" — only runs with matching --model-profile
        scenario_type: str        - "fuzz" or "validation" (for --fuzz-only / --validation-only)

    And implement:
        async def run(self, client: FuzzClient, config: RunConfig) -> list[ScenarioResult]
    """

    name: str = "base"
    description: str = ""
    tags: list[str] = []
    requires_validator: bool = False
    model_filter: str | None = None
    scenario_type: str = "fuzz"

    @abc.abstractmethod
    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        """Execute all tests in this scenario and return results."""
        ...

    # -- helpers available to all scenarios --

    @staticmethod
    def make_result(
        scenario: str,
        test: str,
        verdict: Verdict,
        **kwargs: Any,
    ) -> ScenarioResult:
        return ScenarioResult(
            scenario_name=scenario, test_name=test, verdict=verdict, **kwargs
        )

    @staticmethod
    async def timed_request(
        coro: Awaitable[Any] | Coroutine[Any, Any, Any],
    ) -> tuple[Any, float]:
        """Run a coroutine and return (result, elapsed_ms)."""
        t0 = time.perf_counter()
        result = await coro
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed
