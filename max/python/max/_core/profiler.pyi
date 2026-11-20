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

"""Provides Python bindings for the MAX profiler."""

class Trace:
    """
    A context manager for creating profiling spans.

    Examples:
        >>> with Trace("foo", color="modular_purple"):
        >>>   # Run `bar()` inside the profiling span.
        >>>   bar()
        >>> # The profiling span ends when the context manager exits.

    Args:
        message: The name of the span.
        color: The color of the span.
    """

    def __init__(self, message: str, color: str = "modular_purple") -> None: ...
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

def load_profiler_plugin() -> bool:
    """
    Loads the profiler plugin if it can be found.

    On AMD, call it before any device is created; loaded later, the plugin
    can't record.

    Returns:
        True if a profiler is available.
    """

def start_capture() -> bool:
    """
    Starts a libkineto capture if the profiler is idle.

    Call it on a thread with a current device context, such as the thread
    that created the devices; otherwise the capture doesn't start until a
    device is next initialized. The capture runs until :func:`stop_capture`,
    ignoring the session's profiling step-window settings. Blocks while a
    :func:`stop_capture` is saving.

    Returns:
        True if this call started a capture. False if no profiler is
        available, or if a capture or a Dynolog on-demand trace is already
        running, which this call leaves alone.
    """

def stop_capture(output_path: str) -> str:
    """
    Stops the capture :func:`start_capture` started, saving its trace.

    Leaves any other capture running, such as one the session's profiling
    config started. May be called from any thread.

    Args:
        output_path: File to write the trace to, expanded like
            ``MODULAR_MAX_DEBUG_PROFILING_OUTPUT_PATH``.

    Returns:
        An empty string on success, otherwise why the trace is missing or
        incomplete.
    """

def request_burst(reason: str) -> bool:
    """
    Requests an extra step-window burst for an event, opened now.

    Applies only to a capture with ``profiling-period-steps`` set, where it
    adds a burst between scheduled ones for an event the schedule would
    likely miss, such as a prefill batch. Call it just before the execute
    the burst should start with. The burst covers ``profiling-active-steps``
    steps and is saved to ``<output>.event<k>.json``, labeled with
    ``reason``. At most one request is honored per gap between scheduled
    bursts, and none before the first scheduled burst.

    Args:
        reason: Why the burst was requested, recorded in the trace.
            Characters outside ``[A-Za-z0-9_.-]`` become ``_``.

    Returns:
        ``True`` if a burst opened. ``False`` when profiling is off, which
        costs a single check, or when the request was not honored.
    """

def is_capture_recording() -> bool:
    """
    Returns whether a capture is recording.

    Covers a capture from :func:`start_capture` or the session's profiling
    config.
    """
