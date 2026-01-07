# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Tests for bencher_utils.py - Bench class and throughput calculations."""

import csv
import sys
import tempfile
from io import StringIO
from pathlib import Path

import pytest
from bencher_utils import Bench, ThroughputMeasure


def _dump_report_to_dict(bench: Bench) -> dict[str, str]:
    """Run dump_report and return header->value mapping."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "output.csv"
        old_stdout, sys.stdout = sys.stdout, StringIO()
        try:
            bench.dump_report(output_path)
        finally:
            sys.stdout = old_stdout
        with output_path.open() as f:
            reader = csv.reader(f)
            headers, values = next(reader), next(reader)
        return dict(zip(headers, values, strict=False))


def test_met_label_indicates_seconds() -> None:
    """MET_LABEL should say seconds (s), not milliseconds (ms)."""
    assert "(s)" in Bench.MET_LABEL
    assert "(ms)" not in Bench.MET_LABEL


@pytest.mark.parametrize(
    "bytes_transferred,elapsed_sec,expected_gbps",
    [
        (1_000_000_000, 1.0, 1.0),  # 1 GB in 1s = 1 GB/s
        (100_000_000, 0.1, 1.0),  # 100 MB in 0.1s = 1 GB/s
        (10_000_000, 50e-6, 200.0),  # 10 MB in 50µs = 200 GB/s
    ],
)
def test_throughput_compute(
    bytes_transferred: int, elapsed_sec: float, expected_gbps: float
) -> None:
    """ThroughputMeasure.compute() correctly calculates GB/s from seconds."""
    measure = ThroughputMeasure(metric=Bench.bytes, value=bytes_transferred)
    assert measure.compute(elapsed_sec) == pytest.approx(
        expected_gbps, rel=1e-6
    )


def test_dump_report_time_and_throughput() -> None:
    """dump_report outputs time in seconds and computes throughput correctly."""
    bench = Bench(
        name="test",
        met=1.0,  # 1 second
        iters=1,
        metric_list=[
            ThroughputMeasure(metric=Bench.bytes, value=1_000_000_000)
        ],
    )
    result = _dump_report_to_dict(bench)

    # Time should be in seconds
    assert float(result["met (s)"]) == pytest.approx(1.0, rel=1e-6)
    # 1 GB in 1 second = 1 GB/s
    assert float(result["DataMovement (GB/s)"]) == pytest.approx(1.0, rel=1e-6)
