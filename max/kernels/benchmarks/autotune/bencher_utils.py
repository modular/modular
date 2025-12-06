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

import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchMetric:
    code: int
    """Op-code of the Metric."""
    name: str
    """Metric's name."""
    unit: str
    """Metric's throughput rate unit (count/second)."""


@dataclass
class ThroughputMeasure:
    """Records a throughput metric of metric BenchMetric and value."""

    metric: BenchMetric
    """Type of throughput metric."""
    value: int
    """Measured count of throughput metric."""

    def compute(self, elapsed_sec: float) -> float:
        """Computes throughput rate for this metric per unit of time (second).

        Args:
            elapsed_sec: Elapsed time measured in seconds.

        Returns:
            The throughput values as a floating point 64.
        """
        # TODO: do we need support other units of time (ms, ns)?
        return (self.value) * 1e-9 / elapsed_sec


@dataclass
class Bench:
    name: str
    iters: int
    met: float

    metric_list: list[ThroughputMeasure] = field(default_factory=list)

    elements = BenchMetric(0, "throughput", "GElems/s")
    bytes = BenchMetric(1, "DataMovement", "GB/s")
    flops = BenchMetric(2, "Arithmetic", "GFLOPS/s")
    theoretical_flops = BenchMetric(3, "TheoreticalArithmetic", "GFLOPS/s")

    BENCH_LABEL = "name"
    ITERS_LABEL = "iters"
    MET_LABEL = "met (ms)"

    def dump_report(self, output_path: Path) -> None:
        output: list[str] = []

        metrics = [
            f"{m.metric.name} ({m.metric.unit})" for m in self.metric_list
        ]
        s = [self.BENCH_LABEL, self.ITERS_LABEL, self.MET_LABEL] + metrics
        output += [", ".join(s)]

        metric_vals = [
            f"{m.compute(self.met * 1e-3)}" for m in self.metric_list
        ]
        vals = [self.name, self.iters, self.met] + metric_vals
        output += [", ".join([str(v) for v in vals])]

        output_str = "\n".join(output)
        with open(output_path, "w") as f:
            f.write(output_str + "\n")

        print(output_str)


def arg_parse(handle: str, default: str = "", short_handle: str = "") -> str:
    # TODO: add constraints on dtype of return value

    handle = handle.lstrip("-")
    short_handle = short_handle.lstrip("-")
    args = sys.argv
    for i in range(len(args)):
        if handle and args[i].startswith("--" + handle):
            if "=" in args[i]:
                name_val = args[i].split("=")
                return name_val[1]
            else:
                return args[i + 1]
        elif short_handle and args[i].startswith("-" + short_handle):
            return args[i + 1]
    return default
