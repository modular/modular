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
"""This module provides GPU profiling functionality.

The profiler module enables performance profiling of GPU code blocks through a simple
context manager interface. It includes:

- ProfileBlock: A context manager for timing code blocks
- Configurable profiling that can be enabled/disabled at compile time
- Nanosecond precision timing using perf_counter_ns()
- Source location tracking for profiled blocks
- Formatted timing output

Example:

```mojo
from gpu import profiler
    with profiler.ProfileBlock("my_kernel"):
        # Code to profile
        run_gpu_kernel()
```
"""

from time import perf_counter_ns

from builtin._location import __call_location, _SourceLocation
from builtin.io import _printf


@fieldwise_init
struct ProfileBlock[enabled: Bool = False](Copyable, Movable):
    """A struct for profiling code blocks.

    This struct provides context manager functionality to profile code blocks.
    When enabled, it records the start and end time of the block and prints
    the timing information.

    Parameters:
        enabled: Whether profiling is enabled for this block.
    """

    var name: StaticString
    """Name of the profiling block used for identification in timing output."""

    var loc: _SourceLocation
    """Source code location information for the profiling block, including file, line, and column."""

    var start_time: UInt
    """Start time of the profiling block in nanoseconds, captured using perf_counter_ns()."""

    @always_inline
    @implicit
    fn __init__(out self, name: StaticString):
        """Initialize a new ProfileBlock.

        Args:
            name: Name to identify this profiling block.
        """
        self.start_time = 0

        @parameter
        if enabled:
            self.name = name
            self.loc = __call_location()
        else:
            self.name = ""
            self.loc = _SourceLocation(0, 0, "")

    @always_inline
    fn __enter__(mut self):
        """Enter the profiling block and record start time if enabled."""

        @parameter
        if not enabled:
            return
        self.start_time = perf_counter_ns()

    @always_inline
    fn __exit__(mut self):
        """Exit the profiling block, record end time and print timing if enabled.
        """

        @parameter
        if not enabled:
            return

        var end_time = perf_counter_ns()

        _printf["@ %.*s %ld\n"](
            len(self.name), self.name.unsafe_ptr(), self.start_time - end_time
        )
