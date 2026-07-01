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

    Subscribes to CUPTI activity callbacks. Tracy and libkineto are
    mutually exclusive at build time: ``--config=tracy`` builds do not
    link libkineto, so this call is a no-op there. On default builds
    without libkineto (today: macOS and Linux aarch64) or hosts without
    a live CUDA primary context, this is also a safe no-op.
    """

def kineto_disable() -> None:
    """
    Disable the profiler.

    Flushes the trace. On ``--config=tracy`` or non-Linux-x86_64 builds
    where libkineto isn't linked, this is a no-op.
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
