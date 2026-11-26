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

class TraceActivity:
    """
    One activity of a capture: a host range, a runtime call, or device work.

    The fields are the trace file's, except that times are in nanoseconds.
    """

    @property
    def name(self) -> str: ...
    @property
    def category(self) -> str:
        """
        The activity's type, as the trace file's ``cat`` field names it.

        For example ``user_annotation``, ``kernel`` or ``gpu_memcpy``.
        """

    @property
    def start_ns(self) -> int:
        """Start time, in nanoseconds since the Unix epoch."""

    @property
    def duration_ns(self) -> int: ...
    @property
    def device(self) -> int:
        """The device for device work; the process for host activities."""

    @property
    def resource(self) -> int:
        """The stream for device work; the thread track for host activities."""

    @property
    def thread_id(self) -> int: ...
    @property
    def correlation_id(self) -> int: ...
    @property
    def external_id(self) -> int:
        """
        The ID of the range the activity is correlated to, or 0 if none.

        The trace file stores it as the activity's ``External id``.
        """

def stop_capture_activities(
    output_path: str | None = None,
) -> tuple[list[TraceActivity], dict[str, str], str]:
    """
    Stops the capture :func:`start_capture` started, returning its trace.

    As :func:`stop_capture`, but the trace comes back in memory, so it can
    be read without a trace file. Device work is included only for the
    vendors the profiler records.

    Args:
        output_path: Where to also save the trace, expanded like
            ``MODULAR_MAX_DEBUG_PROFILING_OUTPUT_PATH``, or None to save
            nothing.

    Returns:
        The trace's activities; its top-level metadata, each value raw
        JSON, such as ``MAX dropped range spans``; and an empty string on
        success, otherwise why the trace is missing or incomplete.
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
