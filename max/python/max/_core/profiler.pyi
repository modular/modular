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

    Suitable for use on the hot path to elide expensive trace-name
    construction when off.
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
    Return ``True`` iff the profiler was compiled with libkineto support.

    Currently returns ``False`` in every build configuration: the
    function's defining translation unit lives in ``//Support:Profiling``,
    which does not link libkineto, so the recording paths in ``start()`` /
    ``stop()`` are compiled out — even in ``--config=kineto`` builds on
    Linux x86_64, the only configuration that links libkineto into
    ``max._core`` at all (through the not-yet-wired ``:ProfilingKineto``
    backend; default builds carry no libkineto).  Starts returning ``True``
    once the recording path is wired through that backend.  The Python
    integration
    tests use this to skip end-to-end file-creation assertions whenever
    ``stop()`` cannot actually produce a trace.
    """

def kineto_can_record() -> bool:
    """
    Return ``True`` iff this process can actually record a trace right now.

    Stricter than :func:`kineto_have_libkineto` (and therefore likewise
    ``False`` in every current build configuration): also requires that a
    CUDA primary context is bound on the calling thread.  Without one,
    ``enable()`` skips ``libkineto.prepareTrace`` / ``startTrace`` and
    ``disable()`` symmetrically skips trace serialization, so no file is
    produced.

    Used by ``test_kineto_profiling.py`` to skip end-to-end file-creation
    and ``ProfilingError`` assertions on hosts that cannot record (no
    libkineto in the build, or no live CUDA context — e.g. CI runners
    without NVIDIA hardware).
    """
