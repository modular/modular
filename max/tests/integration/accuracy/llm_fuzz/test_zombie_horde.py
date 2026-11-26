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

import asyncio

import pytest
from client import FuzzClient, RawResponse, RunConfig
from scenarios import BaseScenario, ScenarioResult, Verdict, zombie_horde


class _Finishes(BaseScenario):
    name = "finishes"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        return [self.make_result(self.name, "own_verdict", Verdict.FAIL)]


class _Hangs(BaseScenario):
    name = "hangs"

    async def run(
        self, client: FuzzClient, config: RunConfig
    ) -> list[ScenarioResult]:
        await asyncio.sleep(60)
        return []


class _HealthyClient(FuzzClient):
    async def health_check(self) -> RawResponse:
        return RawResponse(status=200, headers={}, body="", elapsed_ms=0.0)


def _run_horde(
    monkeypatch: pytest.MonkeyPatch,
    scenarios: dict[str, type[BaseScenario]],
    timeout_sec: float,
) -> dict[str, Verdict]:
    monkeypatch.setattr(zombie_horde, "get_all_scenarios", lambda: scenarios)
    config = RunConfig(
        base_url="http://unused", endurance_duration_sec=timeout_sec
    )
    results = asyncio.run(
        zombie_horde.ZombieHorde().run(_HealthyClient(config), config)
    )
    return {r.test_name: r.verdict for r in results}


def test_finished_wave_keeps_results_and_checks_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verdicts = _run_horde(monkeypatch, {"finishes": _Finishes}, 0.2)

    assert verdicts["finishes__own_verdict"] == Verdict.FAIL
    assert verdicts["wave_1_cycle_1_health_check"] == Verdict.PASS


def test_wave_timeout_reports_only_unfinished_scenarios(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verdicts = _run_horde(
        monkeypatch, {"finishes": _Finishes, "hangs": _Hangs}, 0.5
    )

    assert verdicts["finishes__own_verdict"] == Verdict.FAIL
    assert verdicts["hangs__horde_timeout"] == Verdict.ERROR
    assert "finishes__horde_timeout" not in verdicts
