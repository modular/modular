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

"""Records a suite under virtual devices, then replays it, and checks both.

Every run is a subprocess: virtual-device mode latches process-wide at the
first device creation, so the recording process cannot also be the replaying
one, nor the process running these assertions.

The suite below is a GPU test in miniature -- one test that loads and executes,
one that reaches for hardware before compiling anything, one that opts out, one
that compiles two graphs a name and a signature cannot tell apart. Its graphs
are CPU graphs and it asks torch rather than MAX for the accelerator it does not
have, so it runs on any host: what is under test is which compiles were recorded
and what replay does with them, not the arch they were compiled for. Presenting
a virtual accelerator is `max/internal/driver/test/test_virtual_device_hal.py`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

# A test the record run never saw, and a graph shape it never compiled: the two
# ways the consuming run can diverge from the producing one. Both are driven
# from the environment so that the two runs share one file, and so the node ids
# a divergent run reports are the ones the record run wrote down.
_SUITE = """
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType

WIDTH = int(os.environ.get("SUITE_WIDTH", "4"))
ADDEND = float(os.environ.get("SUITE_ADDEND", "1.0"))


def _graph(addend: float) -> Graph:
    dtype = TensorType(DType.float32, [WIDTH], device=DeviceRef.CPU())
    with Graph("test", input_types=[dtype]) as graph:
        graph.output(graph.inputs[0].tensor + addend)
    return graph


def test_loads_and_executes() -> None:
    model = InferenceSession(devices=[CPU()]).load(_graph(ADDEND))
    outputs = model.execute(np.zeros(WIDTH, dtype=np.float32))
    np.testing.assert_allclose(outputs[0].to_numpy(), np.full(WIDTH, ADDEND))


def test_stages_a_device_tensor_first() -> None:
    torch.zeros(WIDTH, device="cuda")
    InferenceSession(devices=[CPU()]).load(_graph(1.0))


@pytest.mark.compile_on_gpu("pinned by the plugin's own test")
def test_opts_out() -> None:
    InferenceSession(devices=[CPU()]).load(_graph(3.0))


def test_compiles_two_graphs_of_one_name() -> None:
    session = InferenceSession(devices=[CPU()])
    session.load(_graph(1.0))
    session.load(_graph(2.0))


if os.environ.get("SUITE_EXTRA"):

    def test_added_after_the_record() -> None:
        InferenceSession(devices=[CPU()]).load(_graph(5.0))
"""


@dataclass
class _Run:
    """One pytest subprocess, and what it left behind."""

    returncode: int
    output: str
    artifacts: Path

    def record(self) -> dict[str, object]:
        """Returns the account the record run wrote of itself."""
        return json.loads((self.artifacts / "record.json").read_text())

    def manifest(self) -> dict[str, object]:
        """Returns the manifest naming the artifacts it exported."""
        return json.loads((self.artifacts / "manifest.json").read_text())

    def outcomes(self) -> dict[str, str]:
        """Returns each recorded test's outcome, keyed by its name."""
        tests = self.record()["tests"]
        assert isinstance(tests, dict)
        return {
            nodeid.partition("::")[2]: entry["outcome"]
            for nodeid, entry in tests.items()
        }


def _run(
    suite: Path, artifacts: Path, environment: dict[str, str], scratch: Path
) -> _Run:
    """Runs ``suite`` in a subprocess with the plugin loaded.

    Args:
        suite: The test file to run.
        artifacts: Where the record run writes, or the replay run reads.
        environment: The plugin's own variables, which pick the mode.
        scratch: A directory for MAX's compile caches.

    Returns:
        What the run returned and left behind.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            # Absolute, and so is the node id pytest derives from it. Both
            # runs are started from this process's working directory, which is
            # what makes the record readable by the replay.
            str(suite),
            "-p",
            "precompile_mefs_plugin",
            "-p",
            "no:cacheprovider",
            "-v",
        ],
        capture_output=True,
        text=True,
        env=os.environ
        | {
            # The subprocess has to import what this test imports, and find
            # the plugin by the bare module name the `-p` above uses.
            "PYTHONPATH": os.pathsep.join(sys.path),
            "MODULAR_DERIVED_PATH": str(scratch),
        }
        | environment,
    )
    return _Run(
        returncode=completed.returncode,
        output=completed.stdout + completed.stderr,
        artifacts=artifacts,
    )


@pytest.fixture(scope="module")
def suite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("suite") / "test_recorded_suite.py"
    path.write_text(_SUITE)
    return path


@pytest.fixture(scope="module")
def recorded(
    suite: Path, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[_Run]:
    artifacts = tmp_path_factory.mktemp("artifacts")
    run = _run(
        suite,
        artifacts,
        {
            "PRECOMPILE_MEFS_MODE": "record",
            "PRECOMPILE_MEFS_RECORD_DIR": str(artifacts),
            # An accelerator this host does not have, which is the point: a
            # record run compiles for one that is not attached.
            "PRECOMPILE_MEFS_ACCELERATOR": "cuda:sm_90a",
            # Left to the host, so the artifacts this exports can be
            # initialized by the replay run below.
            "PRECOMPILE_MEFS_CPU_TARGET": "",
        },
        tmp_path_factory.mktemp("record_scratch"),
    )
    yield run


def _replay(
    suite: Path,
    recorded: _Run,
    tmp_path_factory: pytest.TempPathFactory,
    extra: dict[str, str],
) -> _Run:
    """Replays ``suite`` against what the record run left behind.

    Args:
        suite: The test file to run.
        recorded: The record run whose artifacts to replay.
        tmp_path_factory: For the compile-cache scratch directory.
        extra: What to change about the suite, to diverge from the record.

    Returns:
        The replay run.
    """
    return _run(
        suite,
        recorded.artifacts,
        {
            "PRECOMPILE_MEFS_MODE": "replay",
            # An absolute path, which the runfiles resolver passes through.
            "PRECOMPILE_MEFS_REPLAY_RLOCATIONS": str(recorded.artifacts),
        }
        | extra,
        tmp_path_factory.mktemp("replay_scratch"),
    )


def test_record_exits_zero_although_no_test_could_finish(
    recorded: _Run,
) -> None:
    # The verdicts belong to the run on the GPU. This one is a build action,
    # and a build action that fails stops the test from ever running.
    assert recorded.returncode == 0, recorded.output


def test_record_classifies_each_test(recorded: _Run) -> None:
    assert recorded.outcomes() == {
        "test_loads_and_executes": "recorded",
        "test_stages_a_device_tensor_first": "no_compile",
        "test_opts_out": "opted_out",
        "test_compiles_two_graphs_of_one_name": "passed",
    }


def test_record_keeps_two_graphs_of_one_name_apart(recorded: _Run) -> None:
    # Nothing about the two graphs differs except what they compute, so what
    # tells their artifacts apart is each compile's position within the test.
    tests = recorded.record()["tests"]
    assert isinstance(tests, dict)
    exported = next(
        entry["exported"]
        for nodeid, entry in tests.items()
        if "test_compiles_two_graphs_of_one_name" in nodeid
    )
    assert len(set(exported)) == 2
    # The keys agree on graph name, signature and test, and differ only in
    # that position, which is the whole claim.
    assert len({key.rsplit("-", 1)[0] for key in exported}) == 1

    # And the store recorded what the plugin says it handed out.
    graphs = recorded.manifest()["graphs"]
    assert isinstance(graphs, list)
    assert set(exported) <= {entry["key"] for entry in graphs}
    assert len(list(recorded.artifacts.glob("*.mef"))) == 3


def test_replay_initializes_every_recorded_graph(
    suite: Path, recorded: _Run, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The end-to-end claim: a test that recorded a graph passes here without
    # compiling it, which means the artifact initialized and executed.
    replayed = _replay(suite, recorded, tmp_path_factory, {})

    assert "3 graphs answered from artifacts" in replayed.output
    assert "::test_loads_and_executes PASSED" in replayed.output
    assert "::test_compiles_two_graphs_of_one_name PASSED" in replayed.output


def test_replay_compiles_what_the_record_run_could_not(
    suite: Path, recorded: _Run, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The test that reached for hardware first, and the one that opted out:
    # both compile here, and the summary says so rather than leaving a reader
    # to wonder why the GPU time did not drop. Whether they then pass is the
    # host's business -- one of them wants a GPU this may not have.
    replayed = _replay(suite, recorded, tmp_path_factory, {})

    assert "2 tests and 0 fixtures compiled on the GPU" in replayed.output
    assert "::test_opts_out" in replayed.output


def test_replay_reports_a_graph_the_record_run_never_compiled(
    suite: Path, recorded: _Run, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # What a code change between the two runs looks like: the same tests build
    # graphs the artifacts do not describe. Quietly recompiling them would make
    # the split look like it was still working.
    replayed = _replay(suite, recorded, tmp_path_factory, {"SUITE_WIDTH": "8"})

    assert replayed.returncode != 0
    assert "no precompiled artifact" in replayed.output
    assert "MEF replay" in replayed.output


def test_replay_reports_a_graph_whose_body_changed(
    suite: Path, recorded: _Run, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # The divergence a name and a signature cannot catch: the same test builds
    # a graph of the same shape computing something else, so the artifact
    # recorded for it is found and is the wrong one. Only the recorded body
    # says so, which is why the plugin's store is built to verify it.
    replayed = _replay(suite, recorded, tmp_path_factory, {"SUITE_ADDEND": "7"})

    assert replayed.returncode != 0
    assert "describes a different graph" in replayed.output
    assert "MEF replay" in replayed.output


def test_replay_reports_a_test_the_record_run_never_saw(
    suite: Path, recorded: _Run, tmp_path_factory: pytest.TempPathFactory
) -> None:
    replayed = _replay(suite, recorded, tmp_path_factory, {"SUITE_EXTRA": "1"})

    assert replayed.returncode != 0
    assert "the record run never reached" in replayed.output
