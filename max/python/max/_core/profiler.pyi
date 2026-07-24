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
    Enable the libkineto-backed profiler.

    Subscribes to CUPTI activity callbacks. On builds without libkineto
    (it is linked only in ``--config=kineto`` builds on Linux x86_64) or
    hosts without a live CUDA primary context, this is a safe no-op.
    """

def kineto_disable() -> None:
    """
    Disable the profiler.

    Flushes the trace. On builds where libkineto isn't linked, this is
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
    Return ``True`` iff the libkineto backend is linked into this process.

    Runtime check on the registered ``:ProfilingKineto`` backend, which is
    linked into ``max._core`` only in ``--config=kineto`` builds on Linux
    x86_64 (#91288) — default builds, including the shipped wheels, carry
    no libkineto and return ``False``, and the recording paths in
    ``start()`` / ``stop()`` are no-ops there.
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
    libkineto in the build, or no live CUDA context — e.g. CI runners
    without NVIDIA hardware).
    """
