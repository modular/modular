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

def kineto_enable() -> None:
    """
    Enable the profiler.

    Loads the profiler plugin (``libMAXProfilerPlugin.so``) on first use
    and subscribes to CUPTI activity callbacks. When the plugin is absent
    or no live CUDA primary context exists, the enable intent is recorded
    but nothing records — a safe no-op.
    """

def kineto_disable() -> None:
    """
    Disable the profiler.

    Flushes the trace. When the profiler plugin is not loaded, this is
    a no-op.
    """

def kineto_wait_for_trace() -> None:
    """
    Block until the most recent disable has finished serializing.

    The Python wrapper in ``InferenceSession.profiling.wait_for_trace``
    surfaces serialization failures as ``ProfilingError`` in a follow-up
    PR; today this binding only blocks.
    """

def kineto_state() -> str:
    """
    Return the current profiler state.

    One of ``"idle"``, ``"warmup"``, ``"active"``, or ``"flushing"``.
    """

def kineto_is_enabled() -> bool:
    """
    Return ``True`` while the profiler is enabled.

    Reflects only the session API's enable intent (``start()`` /
    ``stop()``): it stays ``False`` during Dynolog daemon-driven on-demand
    traces, when ranges do record.  To elide range-annotation work on the
    hot path, gate on :func:`kineto_is_recording` instead.
    """

def kineto_is_recording() -> bool:
    """
    Return ``True`` while a trace is live and ranges record.

    Covers traces of either origin — ``start()`` via the session API or a
    Dynolog daemon-driven on-demand request — so it is the right hot-path
    gate for eliding expensive range-name construction: unlike
    :func:`kineto_is_enabled`, it does not opt the caller out of
    daemon-trace annotation.  Single relaxed atomic load.
    """

def kineto_range_begin(name: str, color: int = 0) -> None:
    """
    Begin a semantic CPU range on the calling thread.

    The range is recorded by libkineto as a Chrome-trace CPU span and
    correlated to the GPU kernels launched while it is open.  When no
    trace is live this is a single predicted branch, so calling it
    unconditionally is safe — but constructing the ``name`` string still
    costs Python-side work; gate on :func:`kineto_is_recording` in tight
    loops (not :func:`kineto_is_enabled`, which stays ``False`` during
    Dynolog daemon-driven traces even though ranges record).

    Must be paired with :func:`kineto_range_end` on the same thread.
    Prefer the ``session.profiling.range(...)`` context manager, which
    guarantees pairing.  Unbalanced begins while a trace is live hold
    memory per call until the per-thread depth cap (2^20), beyond which
    they are dropped.
    """

def kineto_range_end() -> None:
    """
    End the innermost open semantic range on the calling thread.

    Pairing is tracked per-thread in the C++ runtime: an end without a
    matching begin (for example after the profiler was stopped between
    the two calls) is a safe no-op.
    """

def kineto_last_trace_error() -> str:
    """
    Return the most recent trace-serialization error message.

    Empty string on success, or before any disable has run.  Used by
    ``InferenceSession.profiling.wait_for_trace()`` to raise
    :class:`max.engine.ProfilingError` when the configured output path
    was unwritable or libkineto could not serialize the in-memory trace.
    Cleared automatically at the next ``start()``.
    """

def kineto_have_libkineto() -> bool:
    """
    Return ``True`` iff the profiler plugin is loaded into this process.

    Attempts to load ``libMAXProfilerPlugin.so`` (``MODULAR_PROFILER_PLUGIN``
    env var, then next to the host library, then the default ``dlopen``
    search). The plugin ships only with internal builds:
    external wheels return ``False`` unless one is deployed alongside, and
    the recording paths in ``start()`` / ``stop()`` are no-ops there. The
    name predates the plugin split and does not imply a specific backend.
    """

def kineto_can_record() -> bool:
    """
    Return ``True`` iff this process can actually record a trace right now.

    Stricter than :func:`kineto_have_libkineto`: also requires that a
    CUDA primary context is bound on the calling thread.  Without one,
    ``enable()`` skips ``libkineto.prepareTrace`` / ``startTrace`` and
    ``disable()`` symmetrically skips trace serialization, so no file is
    produced.

    Used by ``test_kineto_profiling.py`` to skip end-to-end file-creation
    and ``ProfilingError`` assertions on hosts that cannot record (no
    profiler plugin available, or no live CUDA context — e.g. CI runners
    without NVIDIA hardware).
    """
