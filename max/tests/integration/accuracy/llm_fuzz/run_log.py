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
"""Structured JSONL run logging for llm-fuzz."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from scenarios import ScenarioResult


class RunLog:
    """Writes structured JSONL events to a run log file.

    Defaults to ``logs/run-{timestamp}.jsonl``. Pass ``log_file`` to override
    with an explicit path; the parent directory is created if needed.
    """

    def __init__(self, log_file: str | None = None):
        if log_file is None:
            os.makedirs("logs", exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self._path = os.path.join("logs", f"run-{ts}.jsonl")
        else:
            parent = os.path.dirname(log_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._path = log_file
        self._file = open(self._path, "w")

    def _write(self, event: str, **data: object) -> None:
        record = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def log_run_start(
        self,
        url: str,
        model: str,
        scenario_count: int,
        timeout: float,
        max_concurrency: int,
    ) -> None:
        self._write(
            "run_start",
            url=url,
            model=model,
            scenario_count=scenario_count,
            timeout=timeout,
            max_concurrency=max_concurrency,
        )

    def log_scenario_start(self, scenario_name: str) -> None:
        self._write("scenario_start", scenario_name=scenario_name)

    def log_test_result(self, result: ScenarioResult) -> None:
        self._write(
            "test_result",
            scenario_name=result.scenario_name,
            test_name=result.test_name,
            verdict=result.verdict.value,
            status_code=result.status_code,
            elapsed_ms=result.elapsed_ms,
            detail=result.detail,
            error=result.error,
        )

    def log_scenario_end(
        self,
        scenario_name: str,
        elapsed_ms: float,
        pass_count: int,
        fail_count: int,
    ) -> None:
        self._write(
            "scenario_end",
            scenario_name=scenario_name,
            elapsed_ms=elapsed_ms,
            pass_count=pass_count,
            fail_count=fail_count,
        )

    def log_run_end(self, summary: dict[str, object]) -> None:
        self._write("run_end", **summary)

    def close(self) -> None:
        self._file.close()

    @property
    def path(self) -> str:
        return self._path
