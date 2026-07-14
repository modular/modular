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

"""End-to-end HTA-compatibility contract test for the MAX profiler (MXTOOLS-190).

Produces a libkineto trace from a tiny real CUDA workload, then loads it into
Meta's `Holistic Trace Analysis`_ (HTA) library and asserts the resulting
``TraceAnalysis.get_temporal_breakdown()`` is non-empty. This pins the
cross-tool contract that the rest of the suite only asserts at the JSON-
syntactic level: if libkineto's output schema ever drifts from what HTA
expects, this test fails immediately rather than at the user's analysis step.

The test is gated at three layers (Bazel ``["gpu"]`` + ``//:nvidia_gpu``, the
``accelerator_count()`` / ``kineto_can_record()`` runtime probes, and
``importorskip("hta")``); see the PR description for the rationale behind each.

.. _Holistic Trace Analysis: https://github.com/facebookresearch/HolisticTraceAnalysis
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from max._core.profiler import (
    kineto_can_record,
    kineto_disable,
    kineto_is_enabled,
)
from max.driver import Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_skip_without_accelerator = pytest.mark.skipif(
    not accelerator_count(),
    reason="HTA temporal breakdown requires at least one real GPU kernel.",
)


@pytest.fixture(autouse=True)
def _disable_profiler_after_each_test() -> Iterator[None]:
    """Restore the process-wide profiler to disabled after every test.

    Mirrors the fixture in ``test_kineto_profiling.py``: libkineto's state is
    process-global, so a test that fails mid-capture would otherwise leak an
    enabled profiler into every subsequent test in the same pytest run.
    """
    yield
    if kineto_is_enabled():
        kineto_disable()


def _build_tiny_add_graph() -> Graph:
    """A two-input add graph on GPU(0) — the smallest workload that exercises
    a real CUDA kernel launch, which is what HTA's temporal breakdown needs
    in order to classify time as ``compute`` rather than ``idle``.
    """
    input_type = TensorType(
        dtype=DType.float32,
        shape=["batch", "channels"],
        device=DeviceRef.GPU(0),
    )
    with Graph("hta_smoke_add", input_types=(input_type, input_type)) as graph:
        out = ops.add(graph.inputs[0], graph.inputs[1])
        graph.output(out)
    return graph


@_skip_without_accelerator
def test_libkineto_trace_is_hta_consumable(tmp_path: Path) -> None:
    """A libkineto trace produced by ``session.profiling`` round-trips into HTA.

    The end-to-end chain is the load-bearing assertion: if any link
    (libkineto schema, JSON layout, HTA importer) drifts, this fails. The
    actual breakdown values are workload-dependent and intentionally not
    asserted on — what matters is that HTA recognized the file and produced
    *some* analysis.
    """
    # importorskip on the submodule, not the top-level package: importing
    # bare ``hta`` does not guarantee ``hta.trace_analysis`` is an attribute
    # (nor that a future reorganization keeps it one), and a plain attribute
    # access after a top-level importorskip would ERROR the test rather than
    # skip it.
    hta_trace_analysis = pytest.importorskip("hta.trace_analysis")
    TraceAnalysis = hta_trace_analysis.TraceAnalysis

    trace_path = tmp_path / "hta-smoke-trace.json"

    # Construct the session first: building an ``Accelerator`` binds device 0's
    # CUDA primary context, which is the precondition ``kineto_can_record()``
    # probes. Checking recording capability *here*, after the context exists,
    # matters once recording is live — a collection-time ``skipif`` runs
    # before any session exists and would skip even on a capable runner.
    # Today the check is False everywhere regardless: the recording path is
    # not yet wired through the libkineto backend (and post-#91288 only
    # ``--config=kineto`` builds on Linux x86_64 link libkineto at all), so
    # this test skips in every configuration until the wiring lands.
    session = InferenceSession(devices=[Accelerator()])
    if not kineto_can_record():
        pytest.skip(
            "libkineto recording path inactive — recording is not wired into "
            "this build (libkineto requires --config=kineto on Linux x86_64), "
            "or no CUDA primary context is bound."
        )

    # ``session.debug`` is a process-global singleton; snapshot and restore the
    # output path so this test can't leak its trace location into any sibling.
    original_output_path = session.debug.profiling_output_path
    session.debug.profiling_output_path = str(trace_path)
    try:
        model = session.load(_build_tiny_add_graph())

        a = np.ones((4, 8), dtype=np.float32)
        b = np.full((4, 8), 2.0, dtype=np.float32)

        session.profiling.start()
        # A single ``execute`` is enough for HTA's temporal breakdown to have
        # something to classify; the goal is contract validation, not workload
        # coverage.
        model.execute(a, b)
        session.profiling.stop()
        session.profiling.wait_for_trace()
    finally:
        session.debug.profiling_output_path = original_output_path

    assert trace_path.exists(), (
        f"libkineto did not write the expected trace at {trace_path!s}"
    )

    # HTA's TraceAnalysis expects a directory containing one or more trace
    # files — it discovers them by glob, not by path. Pointing at tmp_path
    # gives it our single file in isolation.
    analyzer = TraceAnalysis(trace_dir=str(tmp_path))
    # ``visualize=False`` keeps HTA on the parse→DataFrame path we assert on and
    # skips plotly figure rendering — wasted work on a headless CI runner.
    breakdown = analyzer.get_temporal_breakdown(visualize=False)

    # An empty breakdown means HTA failed to parse any events — either a
    # schema mismatch or a malformed JSON envelope. A non-empty one means
    # the full chain (libkineto JSON writer → HTA parser → DataFrame) is
    # intact; we deliberately don't assert on column values because those
    # depend on hardware, driver version, and CUPTI activity records.
    assert not breakdown.empty, (
        "HTA produced an empty temporal breakdown for the libkineto trace; "
        "likely a schema drift between the libkineto SHA pinned in "
        "MODULE.bazel and the installed HTA version."
    )
