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

"""MAX profiler Python bindings."""

class Trace:
    """
    Context manager for creating profiling spans.

    Examples:
        >>> with Trace("foo", color="modular_purple"):
        >>>   # Run `bar()` inside the profiling span.
        >>>   bar()
        >>> # The profiling span ends when the context manager exits.
    """

    def __init__(self, message: str, color: str = "modular_purple") -> None:
        """
        Constructs and initializes the underlying Mojo Trace object.

        Args:
            message: Name of the span.
            color: Color of the span.
        """

    def __enter__(self) -> Trace:
        """Begins a profiling event."""

    def __exit__(
        self,
        exc_type: object | None = None,
        exc_value: object | None = None,
        traceback: object | None = None,
    ) -> None:
        """Ends a profiling event."""

    def mark(self) -> None:
        """Marks an event in the trace timeline."""

def is_profiling_enabled() -> bool:
    """Returns whether profiling is enabled."""

def set_gpu_profiling_state(arg: str, /) -> None:
    """Sets the GPU profiling state."""

def range_begin_with_id(correlation_id: int, name: str) -> int:
    """
    Begins a CPU range whose libkineto ``External id`` is ``correlation_id``.

    Kernels launched on this thread inside the range, outside any nested
    range, carry the ID. Work launched from other threads, such as AsyncRT
    workers, relates to it only by time. The ID is stored in 32 bits, and
    libkineto omits an ID of 0.

    Pass the result to :func:`range_end` on the same thread.

    Args:
        correlation_id: Caller-supplied join key, such as a batch ID.
        name: The range's label in the trace.

    Returns:
        A nonzero range ID, or 0 if no libkineto trace is live, in which
        case nothing is recorded.
    """

def range_end(range_id: int) -> None:
    """
    Ends a range that :func:`range_begin_with_id` opened.

    Does nothing for an ID of 0. Otherwise warns and ends nothing unless
    the range is the innermost one :func:`range_begin_with_id` opened on
    this thread. Ranges opened inside it by other APIs must end first.

    Args:
        range_id: The ID :func:`range_begin_with_id` returned.
    """
