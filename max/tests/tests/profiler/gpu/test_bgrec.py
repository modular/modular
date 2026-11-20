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

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path

import pytest
from max.profiler.gpu import BackgroundRecorder
from max.profiler.gpu.bgrec import _record_samples


def test_smoke() -> None:
    with BackgroundRecorder() as recorder1:
        time.sleep(3)
    assert len(recorder1.stats) > 0

    with BackgroundRecorder(interval=0.1) as recorder2:
        time.sleep(3)
    assert len(recorder2.stats) > len(recorder1.stats)


class _FailingStopProcess:
    """Stands in for a recorder process whose ``stop()`` raises."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.killed = False

    def stop(self) -> Path:
        raise self.error

    def stop_without_request(self) -> None:
        pass

    def kill_gracefully(self) -> None:
        self.killed = True


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("recorder died"),
        BrokenPipeError("broken pipe"),
        TimeoutError("recorder hung"),
    ],
)
def test_exit_records_no_stats_when_recorder_fails(
    error: BaseException, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = BackgroundRecorder()
    proc = _FailingStopProcess(error)
    recorder._proc = proc  # type: ignore[assignment]
    recorder.__exit__(None, None, None)
    assert recorder.stats == []
    assert proc.killed
    assert "GPU stats unavailable" in capsys.readouterr().err


def test_exit_reraises_unexpected_errors() -> None:
    recorder = BackgroundRecorder()
    proc = _FailingStopProcess(ValueError("unexpected"))
    recorder._proc = proc  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unexpected"):
        recorder.__exit__(None, None, None)
    assert proc.killed


class _StopLoop(BaseException):
    """Ends the recorder loop; escapes its ``except Exception`` handler."""


class _FlakyDiag:
    """Fails every other ``get_stats()`` call, then stops the loop."""

    def __init__(self, total_calls: int) -> None:
        self.calls = 0
        self.total_calls = total_calls

    def get_stats(self) -> dict[str, int]:
        self.calls += 1
        if self.calls > self.total_calls:
            raise _StopLoop
        if self.calls % 2 == 0:
            raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")
        return {"sample": self.calls}


def test_recorder_drops_failed_samples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    diag = _FlakyDiag(total_calls=6)
    output = io.BytesIO()
    with pytest.raises(_StopLoop):
        asyncio.run(_record_samples(diag, output, 0))  # type: ignore[arg-type]
    samples = [json.loads(line) for line in output.getvalue().splitlines()]
    assert samples == [{"sample": 1}, {"sample": 3}, {"sample": 5}]
    assert capsys.readouterr().err.count("GPU stats sample dropped") == 1
