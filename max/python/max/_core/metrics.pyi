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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""Provides in-process metrics collection for the Modular runtime."""

class HistogramSnapshot:
    """
    Statistics accumulated by a histogram instrument between :func:`collect()` calls.

    All fields reflect data points recorded since the previous
    :func:`collect()`.
    """

    @property
    def count(self) -> int:
        """The number of data points recorded."""

    @property
    def sum(self) -> float:
        """The sum of all recorded values."""

    @property
    def min(self) -> float:
        """The minimum recorded value."""

    @property
    def max(self) -> float:
        """The maximum recorded value."""

class CounterSample:
    """
    A single counter reading from a :func:`collect()` snapshot.

    Counter values are cumulative and never reset.
    """

    @property
    def name(self) -> str:
        """The instrument name."""

    @property
    def value(self) -> int:
        """The cumulative count."""

class GaugeSample:
    """A single gauge value from an instantaneous read."""

    @property
    def name(self) -> str:
        """The instrument name."""

    @property
    def value(self) -> int:
        """The current gauge value."""

class HistogramSample:
    """
    A single histogram reading from a :func:`collect()` snapshot.

    The underlying accumulator is reset on each :func:`collect()` call, so
    :attr:`ss` reflects only the interval since the previous
    :func:`collect()`.
    """

    @property
    def name(self) -> str:
        """The instrument name."""

    @property
    def ss(self) -> HistogramSnapshot:
        """The statistics for this collection interval."""

class CollatedMetrics:
    """
    A snapshot of all registered instruments, returned by :func:`collect()`.
    """

    @property
    def counters(self) -> list[CounterSample]:
        """The counter readings, one entry per registered counter."""

    @property
    def gauges(self) -> list[GaugeSample]:
        """The gauge readings, one entry per registered gauge."""

    @property
    def histograms(self) -> list[HistogramSample]:
        """The histogram readings, one entry per registered histogram."""

def collect() -> CollatedMetrics:
    """
    Snapshots all registered instruments.

    Histogram accumulators are cleared on each call; counter and gauge
    values are cumulative.

    Returns:
        A :class:`CollatedMetrics` containing the current reading of every
        counter and gauge, and a reset snapshot of every histogram.
    """
