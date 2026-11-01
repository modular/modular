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
"""Verifies that the EP shared expert runs on a side stream and overlaps there.

With ``MODULAR_OVERLAP_SHARED_EXPERT`` on, the default for every EP model,
``forward_moe_sharded_layers`` binds the shared expert to a side stream so it
overlaps the routed-expert dispatch/combine. This test checks that at
execution time. It runs the EP MoE forward pass under ``rocprofv3`` once with
the overlap enabled and once with it disabled, and requires that on every
device the enabled arm adds exactly one stream over the control and that the
stream's kernels run concurrently with other work at least once. Reading the
compiled graph cannot establish either: the ``mo.sequence`` region is there
whether or not the runtime honors it.

Two properties make the result readable, and the second is the one that is
easy to omit:

1. **Uncaptured.** A captured device graph replays as a single launch and its
   kernels are attributed to the launch stream, so a per-stream histogram of a
   captured run shows one stream per device whether the overlap works or not.
   This harness compiles a bare ``Graph`` through ``InferenceSession.load`` and
   executes it directly; device graph capture is driven by the pipeline layer,
   which is not involved here, so there is nothing to disable.

2. **Controlled.** The ``MODULAR_OVERLAP_SHARED_EXPERT=0`` arm is not
   decoration. Other machinery in the process (EP communication, the runtime's
   own copies) may hold streams of its own, so "two streams exist" is not by
   itself evidence the shared expert moved. The control establishes the
   baseline stream count for everything *except* the overlap, and the claim
   under test is the difference between the arms. If the two arms do not
   differ, this test fails as an instrument failure rather than reporting the
   enabled arm's count as a result.

The EP graph is compiled twice, once per arm. That is deliberate rather than
the per-test-recompilation antipattern: ``MODULAR_OVERLAP_SHARED_EXPERT`` is
read while the graph is being built, so the two arms cannot share a compiled
model. It is also why each arm runs in its own process.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import pytest
from max.driver import accelerator_count
from side_stream_worker import N_DEVICES

_WORKER = Path(__file__).resolve().parent / "side_stream_worker.py"
# The bazel test PATH is pinned to system directories, which a ROCm install
# does not put rocprofv3 in.
_ROCPROFV3 = shutil.which("rocprofv3") or shutil.which(
    "rocprofv3", path="/opt/rocm/bin"
)
_MODULE_START = time.monotonic()

# rocprofv3's kernel trace names the dispatching agent and the stream it was
# dispatched on. `Stream_Id` is preferred; `Queue_Id` is the fallback for
# builds that emit only the queue.
_AGENT_COLUMN = "Agent_Id"
_STREAM_COLUMNS = ("Stream_Id", "Queue_Id")
_NAME_COLUMN = "Kernel_Name"
_START_COLUMN = "Start_Timestamp"
_END_COLUMN = "End_Timestamp"

_SKIP_EXIT_CODE = 2
# Outside bazel there is no outer deadline, so each arm gets a fixed budget.
_STANDALONE_ARM_TIMEOUT_S = 1800
# Held back from bazel's TEST_TIMEOUT so the parse, report and assertion
# messages still run after the second arm.
_TIMEOUT_MARGIN_S = 60


class Dispatch(NamedTuple):
    """One traced kernel dispatch."""

    agent: str
    stream: str
    name: str
    start: int
    end: int


def _normalize(name: str) -> str:
    return name.strip().lstrip("\ufeff").strip('"')


def _parse_kernel_trace(trace: Path) -> list[Dispatch]:
    """Reads one rocprofv3 kernel trace into dispatch records."""
    with trace.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {_normalize(f) for f in (reader.fieldnames or [])}

        stream_column = next(
            (c for c in _STREAM_COLUMNS if c in fields),
            None,
        )
        if stream_column is None:
            raise AssertionError(
                f"{trace}: kernel trace has no stream column; looked for "
                f"{list(_STREAM_COLUMNS)}, found {sorted(fields)}"
            )
        required = {_AGENT_COLUMN, _NAME_COLUMN, _START_COLUMN, _END_COLUMN}
        missing = required - fields
        if missing:
            raise AssertionError(
                f"{trace}: kernel trace missing columns {sorted(missing)}"
            )

        dispatches: list[Dispatch] = []
        for row in reader:
            clean = {_normalize(k): (v or "") for k, v in row.items()}
            dispatches.append(
                Dispatch(
                    agent=clean[_AGENT_COLUMN].strip(),
                    stream=clean[stream_column].strip(),
                    name=clean[_NAME_COLUMN].strip().strip('"'),
                    start=int(clean[_START_COLUMN]),
                    end=int(clean[_END_COLUMN]),
                )
            )
    return dispatches


def _outputs_dir(fallback: Path) -> Path:
    """Returns where diagnostics outlive the test sandbox.

    Bazel publishes ``TEST_UNDECLARED_OUTPUTS_DIR``; a pytest ``tmp_path`` is
    discarded with the sandbox, which on a remote executor loses the evidence.
    """
    return Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", fallback))


def _arm_timeout_s() -> float:
    """Returns the deadline for one arm; the two arms run back to back.

    Under bazel the whole test is killed at ``TEST_TIMEOUT``, whose value
    depends on the target size and the ``--config`` in effect. Splitting what
    remains of it lets a slow arm fail with its own output instead of being
    killed from outside.
    """
    total = os.environ.get("TEST_TIMEOUT")
    if total is None:
        return _STANDALONE_ARM_TIMEOUT_S
    elapsed = time.monotonic() - _MODULE_START
    budget = int(total) - _TIMEOUT_MARGIN_S - elapsed
    if budget <= 0:
        raise AssertionError(
            f"TEST_TIMEOUT={total}s leaves no time for either arm"
        )
    return budget / 2


def _run_arm(
    overlap_enabled: bool, out_dir: Path, outputs: Path, timeout_s: float
) -> list[Dispatch]:
    """Runs the worker under rocprofv3 for one arm and parses its trace."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["MODULAR_OVERLAP_SHARED_EXPERT"] = "1" if overlap_enabled else "0"

    assert _ROCPROFV3 is not None
    command = [
        _ROCPROFV3,
        "--kernel-trace",
        "--output-format",
        "csv",
        f"--output-directory={out_dir}",
        "--",
        sys.executable,
        str(_WORKER),
    ]
    completed = subprocess.run(
        command,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if completed.returncode == _SKIP_EXIT_CODE:
        pytest.skip(f"worker declined to run: {completed.stderr.strip()}")
    if completed.returncode != 0:
        raise AssertionError(
            f"rocprofv3 worker exited {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )

    traces = sorted(out_dir.rglob("*_kernel_trace.csv"))
    if len(traces) != 1:
        raise AssertionError(
            f"{out_dir}: expected one *_kernel_trace.csv, found "
            f"{[str(t) for t in traces]}"
        )
    # Before parsing, so a malformed trace is still published.
    shutil.copy(traces[0], outputs / f"{out_dir.name}_kernel_trace.csv")
    dispatches = _parse_kernel_trace(traces[0])
    if not dispatches:
        raise AssertionError(
            f"{traces[0]}: tracer produced no kernel dispatches; the worker "
            "ran but nothing was recorded, so neither arm is readable"
        )
    return dispatches


def _streams_per_agent(
    dispatches: list[Dispatch],
) -> dict[str, dict[str, int]]:
    """Returns ``{agent: {stream: kernel_count}}``."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for dispatch in dispatches:
        counts[dispatch.agent][dispatch.stream] += 1
    return {agent: dict(streams) for agent, streams in counts.items()}


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merges overlapping intervals; input need not be sorted."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _busy_ns(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge(intervals))


def _intersection_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total wall-clock nanoseconds where both stream timelines are busy."""
    total = 0
    merged_a = _merge(a)
    merged_b = _merge(b)
    i = j = 0
    while i < len(merged_a) and j < len(merged_b):
        start = max(merged_a[i][0], merged_b[j][0])
        end = min(merged_a[i][1], merged_b[j][1])
        if end > start:
            total += end - start
        if merged_a[i][1] < merged_b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _added_streams(
    streams_on: Mapping[str, Mapping[str, int]],
    streams_off: Mapping[str, Mapping[str, int]],
) -> dict[str, list[str]]:
    """Returns, per agent, the streams the enabled arm has and the control lacks."""
    return {
        agent: sorted(on.keys() - streams_off.get(agent, {}).keys())
        for agent, on in streams_on.items()
    }


def _side_stream_overlap(
    enabled: list[Dispatch], side_streams: Mapping[str, str]
) -> dict[str, dict[str, float]]:
    """Per-agent wall-clock time the side stream runs alongside other streams.

    ``side_streams`` maps each agent to the one stream its enabled arm adds
    over the control. That identifies the side stream where "second-busiest"
    does not: other work can hold a busier stream in both arms, as PyTorch
    generating the weights does on GPU 0.
    """
    by_agent: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for dispatch in enabled:
        by_agent[dispatch.agent][dispatch.stream].append(
            (dispatch.start, dispatch.end)
        )

    report: dict[str, dict[str, float]] = {}
    for agent, side_stream in side_streams.items():
        streams = by_agent[agent]
        side = streams[side_stream]
        rest = [
            iv for s, ivs in streams.items() if s != side_stream for iv in ivs
        ]
        busy = _busy_ns(side)
        shared = _intersection_ns(side, rest)
        report[agent] = {
            "overlap_ns": float(shared),
            "side_busy_ns": float(busy),
            "overlap_fraction": (shared / busy) if busy else 0.0,
        }
    return report


@pytest.mark.skipif(
    _ROCPROFV3 is None,
    reason="rocprofv3 is required to attribute kernels to streams",
)
@pytest.mark.skipif(
    accelerator_count() < N_DEVICES,
    reason=f"EP MoE fixture needs {N_DEVICES} accelerators",
)
def test_ep_shared_expert_runs_on_a_side_stream(tmp_path: Path) -> None:
    """The EP shared expert adds a stream on every device and overlaps there.

    Fails in distinguishable ways, and the distinction is the point: the
    instrument failing (the tracer recording nothing or the wrong devices, the
    control not discriminating, or the side stream not being identifiable), a
    device showing no extra stream, and the extra stream never running
    concurrently with other work are separate faults with separate messages.
    """
    outputs = _outputs_dir(tmp_path)
    timeout_s = _arm_timeout_s()
    enabled = _run_arm(
        overlap_enabled=True,
        out_dir=tmp_path / "overlap-on",
        outputs=outputs,
        timeout_s=timeout_s,
    )
    disabled = _run_arm(
        overlap_enabled=False,
        out_dir=tmp_path / "overlap-off",
        outputs=outputs,
        timeout_s=timeout_s,
    )

    streams_on = _streams_per_agent(enabled)
    streams_off = _streams_per_agent(disabled)
    added = _added_streams(streams_on, streams_off)
    overlap = _side_stream_overlap(
        enabled, {agent: s[0] for agent, s in added.items() if len(s) == 1}
    )

    artifact = outputs / "ep_side_stream_report.json"
    artifact.write_text(
        json.dumps(
            {
                "overlap_on": {
                    "streams_per_agent": streams_on,
                    "added_streams": added,
                    "side_stream_overlap": overlap,
                },
                "overlap_off": {"streams_per_agent": streams_off},
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nside-stream trace report: {artifact}")
    for agent in sorted(streams_on.keys() | streams_off.keys()):
        print(
            f"  {agent}: on={sorted(streams_on.get(agent, {}))} "
            f"off={sorted(streams_off.get(agent, {}))} "
            f"side-stream overlap={overlap.get(agent, {})}"
        )

    # The comparison below is per device, so both arms must have traced the
    # same devices, and exactly the ones the EP fixture spans.
    assert streams_on.keys() == streams_off.keys(), (
        "the arms traced different devices, so their stream counts are not "
        f"comparable.\n  on:  {sorted(streams_on)}\n  off: {sorted(streams_off)}"
    )
    assert len(streams_on) == N_DEVICES, (
        f"expected kernels on {N_DEVICES} devices, traced "
        f"{len(streams_on)}: {sorted(streams_on)}"
    )

    # The control. If disabling the overlap does not reduce the stream count
    # on any device, this histogram cannot see the overlap, and the enabled
    # arm's count says nothing either way.
    improved = [
        agent
        for agent, on in streams_on.items()
        if len(on) > len(streams_off[agent])
    ]
    assert improved, (
        "control arm did not discriminate: MODULAR_OVERLAP_SHARED_EXPERT=0 "
        "shows the same number of streams per device as =1, so this "
        "measurement cannot distinguish a working side stream from a dead "
        "one. Treat the instrument as unready rather than reading the "
        f"enabled arm.\n  on:  {streams_on}\n  off: {streams_off}"
    )

    # The claim. Every device gained a stream from the overlap. A delta rather
    # than an absolute count, because other work holds streams in both arms:
    # PyTorch generates the weights on GPU 0, so that device can show two
    # streams with its side stream dead.
    unimproved = {
        agent: {"on": sorted(on), "off": sorted(streams_off[agent])}
        for agent, on in streams_on.items()
        if agent not in improved
    }
    assert not unimproved, (
        "shared-expert overlap is enabled but these devices gained no stream "
        f"over the control: {unimproved}. The control did discriminate on "
        f"{improved}, so the histogram is working and these devices genuinely "
        "ran the shared expert on the default stream."
    )

    # The graph requests one side stream, so a device adding more than one
    # leaves the overlap below unattributable. Stream ids are numbered across
    # the process rather than per device, so a change in the order the arms
    # create streams lands here, as does an unrelated stream only the enabled
    # arm creates.
    ambiguous = {agent: s for agent, s in added.items() if len(s) > 1}
    assert not ambiguous, (
        "these devices added more than one stream over the control, so the "
        f"shared expert's stream cannot be identified: {ambiguous}. Treat the "
        "instrument as unready rather than reading the overlap."
    )

    # The overlap. A second stream is necessary but not sufficient: the
    # fork/join could still serialize the side stream behind the main one.
    # Asserted as "ever concurrent" rather than as a fraction, which would make
    # this a performance test.
    serialized = {a: r for a, r in overlap.items() if r["overlap_ns"] <= 0}
    assert not serialized, (
        "the shared expert reached a side stream on every device, but on "
        f"these it never ran concurrently with other work: {serialized}. The "
        "fork/join is serializing it, so the overlap buys nothing."
    )
