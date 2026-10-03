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
"""Pins that the kernel-capture and trace-carrier modules load only when a
process uses them, so the default startup pays nothing for them."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_CAPTURE = "max.serve.telemetry._kernel_capture"
_CARRIER = "max.serve.telemetry._trace_context"
_TRACES = {
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": "http://localhost:4318/v1/traces"
}
_SENTINEL = "loaded:"


def _loaded(code: str, env: dict[str, str] | None = None) -> set[str]:
    """Runs ``code`` in a fresh interpreter and returns which modules loaded."""
    script = (
        f"{code}\nimport sys\n"
        f"print({_SENTINEL!r}, "
        f"*(m for m in ({_CAPTURE!r}, {_CARRIER!r}) if m in sys.modules))"
    )
    child_env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    child_env.pop("OTEL_SDK_DISABLED", None)
    child_env.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)
    child_env.update(env or {})
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        check=True,
        capture_output=True,
        text=True,
    )
    (line,) = (
        line
        for line in result.stdout.splitlines()
        if line.startswith(_SENTINEL)
    )
    return set(line.split()[1:])


def test_model_worker_loads_neither_by_default() -> None:
    assert _loaded("import max.serve.pipelines.model_worker") == set()


def test_api_server_skips_the_capture_module_by_default() -> None:
    assert _CAPTURE not in _loaded("import max.serve.api_server")


@pytest.mark.parametrize(
    ("tracing", "headers", "expected"),
    [
        pytest.param(True, False, set(), id="tracing-only"),
        pytest.param(False, True, set(), id="headers-without-tracing"),
        pytest.param(True, True, {_CAPTURE, _CARRIER}, id="headers-on"),
    ],
)
def test_worker_loads_them_once_headers_are_on(
    tracing: bool, headers: bool, expected: set[str]
) -> None:
    # No plugin, so the child stays quiet and loads no profiler.
    code = (
        "import max._core.profiler\n"
        "max._core.profiler.load_profiler_plugin = lambda: False\n"
        "from max.serve import scheduler\n"
        "from max.serve.config import Settings\n"
        "from max.serve.pipelines import model_worker\n"
        "from max.serve.telemetry import common\n"
        f"s = Settings(kernel_trace_headers={headers}, disable_telemetry=False)\n"
        "common.configure_tracing(s)\n"
        "model_worker._configure_kernel_capture(s, 'prefill_and_decode')\n"
        "assert scheduler._kernel_capture() is None\n"
    )
    assert _loaded(code, _TRACES if tracing else None) == expected


@pytest.mark.parametrize("role", ["prefill_only", "decode_only"])
def test_disaggregated_worker_skips_them(role: str) -> None:
    # Only the prefill_and_decode scheduler takes a capture. Skipping the
    # module also skips the plugin load.
    code = (
        "import max._core.profiler\n"
        "max._core.profiler.load_profiler_plugin = lambda: False\n"
        "from max.serve.config import Settings\n"
        "from max.serve.pipelines import model_worker\n"
        "from max.serve.telemetry import common\n"
        "s = Settings(kernel_trace_headers=True, disable_telemetry=False)\n"
        "common.configure_tracing(s)\n"
        f"model_worker._configure_kernel_capture(s, {role!r})\n"
    )
    assert _loaded(code, _TRACES) == set()
