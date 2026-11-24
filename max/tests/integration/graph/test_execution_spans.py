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

"""Covers the model_execute span emitted around every C++ forward pass.

SpanGuard itself is covered by SpanGuardTest; these cases pin the call site:
that executing a model produces the pair, that a pass costs exactly one pair,
and that the pair carries the batch id the caller set.

The logger is configured by environment variable and its level is process
wide, so each case runs a subprocess writing JSON records to its own file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

# Builds a graph, loads it, and executes it `passes` times, optionally inside a
# request context. Nothing is asserted here -- the records are the output.
_EXECUTE_SCRIPT = """\
    import numpy as np
    from max._core import request_context
    from max.driver import CPU, Buffer
    from max.dtype import DType
    from max.graph import DeviceRef, Graph, TensorType, ops
    from max.engine import InferenceSession

    passes = {passes}
    batch_id = {batch_id}

    tensor_type = TensorType(DType.float32, [4], device=DeviceRef.CPU())
    with Graph("spans", input_types=[tensor_type, tensor_type]) as graph:
        a, b = graph.inputs
        graph.output(ops.add(a, b))

    device = CPU()
    model = InferenceSession(devices=[device]).load(graph)
    operand = Buffer.from_numpy(np.ones(4, dtype=np.float32)).to(device)

    for _ in range(passes):
        if batch_id is not None:
            request_context.set_batch_id(batch_id)
        try:
            model.execute(operand, operand)
        finally:
            if batch_id is not None:
                request_context.clear_batch_id()

    print("DONE")
"""


def _run_and_collect_spans(
    tmp_path: Path, *, passes: int = 1, batch_id: int | None = None
) -> list[dict[str, object]]:
    """Executes a model in a subprocess and returns the span records it wrote.

    Args:
        tmp_path: Directory to write the log file into.
        passes: How many times to execute the model.
        batch_id: Request context to set around each pass, or None for no
            context.

    Returns:
        The ``span_start`` / ``span_end`` records, in the order written.
    """
    log_file = tmp_path / "records.json"
    env = os.environ.copy()
    env.update(
        {
            "MODULAR_LOG_LEVEL": "INFO",
            "MODULAR_LOG_JSON": "1",
            "MODULAR_LOG_FILE": str(log_file),
            # Records go to the file; keeping them off stdout leaves the
            # subprocess's own output readable when a case fails.
            "MODULAR_LOG_STDOUT": "false",
            "MODULAR_LOG_NO_SUMMARY": "1",
        }
    )

    script = textwrap.dedent(_EXECUTE_SCRIPT).format(
        passes=passes, batch_id=batch_id
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0 and "DONE" in result.stdout, (
        f"Execution subprocess failed (rc={result.returncode}).\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    records = [
        json.loads(line)
        for line in log_file.read_text().splitlines()
        if line.strip()
    ]
    return [r for r in records if r.get("event") in ("span_start", "span_end")]


def test_a_forward_pass_emits_a_model_execute_span(tmp_path: Path) -> None:
    spans = _run_and_collect_spans(tmp_path)

    assert [s["event"] for s in spans] == ["span_start", "span_end"]
    # The operation names the start record only: the end record spends its
    # four key-value pairs on event, span_id, duration_us and the batch id,
    # and the shared span_id is what carries the name across.
    assert spans[0]["operation"] == "model_execute"
    assert "operation" not in spans[1]
    assert spans[0]["span_id"] == spans[1]["span_id"]


def test_the_span_reports_a_plausible_duration(tmp_path: Path) -> None:
    spans = _run_and_collect_spans(tmp_path)

    duration_us = spans[1]["duration_us"]
    assert isinstance(duration_us, int)
    # An elementwise add over four floats cannot take a minute; the ceiling is
    # here to catch an uninitialised or wrapped duration, not to time the pass.
    assert 0 < duration_us < 60_000_000


def test_each_pass_emits_exactly_one_pair(tmp_path: Path) -> None:
    # Guards the volume contract: the guard sits on the one entry point every
    # caller funnels through, so a pass costs two records rather than one per
    # nested invocation.
    spans = _run_and_collect_spans(tmp_path, passes=3)

    assert [s["event"] for s in spans] == ["span_start", "span_end"] * 3
    assert len({s["span_id"] for s in spans}) == 3


def test_the_span_carries_the_request_context(tmp_path: Path) -> None:
    spans = _run_and_collect_spans(tmp_path, batch_id=4242)

    assert [s["batch_id"] for s in spans] == [4242, 4242]


def test_the_span_omits_a_batch_id_without_a_context(tmp_path: Path) -> None:
    spans = _run_and_collect_spans(tmp_path)

    assert all("batch_id" not in s for s in spans)
