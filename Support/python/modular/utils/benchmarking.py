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
from typing import Dict, Iterable, Optional


@dataclass
class CompilePerfDetails:
    """Stores compile time and peak memory usage during compilation.

    TODO(#34416): Write the remaining metrics from all frontends.
    Currently only the PT driver `mtorch` records compile-time details, and it
    only records `load_seconds`.
    """

    # Compile time in seconds.
    frontend_to_mgp: Optional[float] = None
    import_seconds: Optional[float] = None
    load_seconds: Optional[float] = None

    # Peak RSS in gigabytes.
    peak_rss_gb_frontend_to_mgp: Optional[float] = None
    peak_rss_gb_import: Optional[float] = None
    peak_rss_gb_load: Optional[float] = None

    @classmethod
    def from_file(
        cls, time_profile: Optional[Path]
    ) -> Optional["CompilePerfDetails"]:
        """Read compile time and peak RSS from the profiling events file."""
        if not time_profile or not time_profile.exists():
            return CompilePerfDetails()

        with open(time_profile, "r") as profile:
            return CompilePerfDetails.from_lines(profile)

    @classmethod
    def from_lines(cls, time_profile: Iterable[str]) -> "CompilePerfDetails":
        compile_perf = CompilePerfDetails()
        for line in time_profile:
            components = line.split()

            # The driver writes a perf event ending in `compileToBinary`.
            # For example the torch driver writes `torch::compileToBinary`.
            if "compileToBinary" in components[-1]:
                # Convert the perf event from microseconds to seconds.
                load_us = int(components[-2])
                compile_perf.load_seconds = load_us / 10**6

        return compile_perf


@dataclass
class BenchmarkResult:
    """Class to hold benchmark results."""

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
    def from_file(cls, summary_path: Path) -> "BenchmarkResult":
        """
        Parses the given MLPerf benchmarking log and returns results.

        Args:
            summary_path (Path): Path to logfile to be parsed.
        """
        with open(summary_path, "r") as summary:
            return BenchmarkResult.from_lines(summary)

    @classmethod
    def from_lines(cls, summary_lines: Iterable[str]) -> "BenchmarkResult":
        """
        Parses the given MLPerf benchmarking log and returns results.

        Args:
            summary_lines (List[str]): Contents of logfile to be parsed.
        """

        def get_value(val: str):
            return float(val.split(":")[1].strip())

        result = BenchmarkResult()
        for line in summary_lines:
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
