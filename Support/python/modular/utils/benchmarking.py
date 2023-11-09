# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """"
Benchmarking Utility Library
"""

from dataclasses import dataclass, fields
from pathlib import Path

from modular.utils.typing import Dict, Iterable, Optional


@dataclass
class BenchmarkResult:
    """
    Class to hold benchmark results
    """

    min_latency: Optional[float] = None
    max_latency: Optional[float] = None
    mean_latency: Optional[float] = None
    percentile_5000: Optional[float] = None
    percentile_9000: Optional[float] = None
    percentile_9500: Optional[float] = None
    percentile_9700: Optional[float] = None
    percentile_9900: Optional[float] = None
    percentile_9990: Optional[float] = None
    qps: Optional[float] = None

    @classmethod
    def from_file(cls, path: Path):
        """
        Parses the given MLPerf benchmarking log and returns results.

        Args:
            path (Path): Path to logfile to be parsed.
        """
        with open(path, "r") as summary:
            return BenchmarkResult.from_lines(summary)

    @classmethod
    def from_lines(cls, lines: Iterable[str]):
        """
        Parses the given MLPerf benchmarking log and returns results.

        Args:
            lines (List[str]): Contents of logfile to be parsed.
        """

        def get_value(val: str):
            return float(val.split(":")[1].strip())

        result = BenchmarkResult()
        for line in lines:
            if "QPS w/o loadgen" in line:
                result.qps = get_value(line)
            elif "Min latency" in line:
                result.min_latency = get_value(line)
            elif "Max latency" in line:
                result.max_latency = get_value(line)
            elif "Mean latency" in line:
                result.mean_latency = get_value(line)
            elif "50.00 percentile" in line:
                result.percentile_5000 = get_value(line)
            elif "90.00 percentile" in line:
                result.percentile_9000 = get_value(line)
            elif "95.00 percentile" in line:
                result.percentile_9500 = get_value(line)
            elif "97.00 percentile" in line:
                result.percentile_9700 = get_value(line)
            elif "99.00 percentile" in line:
                result.percentile_9900 = get_value(line)
            elif "99.90 percentile" in line:
                result.percentile_9990 = get_value(line)
        return result

    def to_dict(self) -> Dict:
        """Return dictionary representation of dataclass"""
        return {x.name: getattr(self, x.name) for x in fields(self)}
