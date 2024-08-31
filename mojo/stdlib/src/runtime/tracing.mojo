# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides tracing utilities."""

from collections.optional import Optional
from sys import external_call
from sys.param_env import env_get_int, is_defined

from utils import StaticIntTuple
from buffer import NDBuffer


fn build_info_asyncrt_max_profiling_level() -> Optional[Int]:
    @parameter
    if not is_defined["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]():
        return None
    return env_get_int["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]()


# ===----------------------------------------------------------------------===#
# TraceCategory
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TraceType(EqualityComparable):
    """An enum-like struct specifying the type of tracing to perform."""

    alias OTHER = TraceType(0)
    alias AsyncRT = TraceType(1)
    alias MEM = TraceType(2)
    alias MOJO = TraceType(3)

    var value: Int

    @always_inline("nodebug")
    fn __eq__(self, rhs: TraceType) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: TraceType) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self.value != rhs.value


# ===----------------------------------------------------------------------===#
# TraceLevel
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TraceLevel(EqualityComparable):
    """An enum-like struct specifying the level of tracing to perform."""

    alias ALWAYS = TraceLevel(0)
    alias OP = TraceLevel(1)
    alias THREAD = TraceLevel(2)

    var value: Int

    @always_inline("nodebug")
    fn __eq__(self, rhs: TraceLevel) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: TraceLevel) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __le__(self, rhs: TraceLevel) -> Bool:
        """Performs less than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than or equal to `rhs`.
        """
        return self.value <= rhs.value


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn is_profiling_enabled[type: TraceType, level: TraceLevel]() -> Bool:
    """Returns True if the profiling is enabled for that specific type and
    level and False otherwise."""
    alias kProfilingTypeWidthBits = 3

    @parameter
    if level == TraceLevel.ALWAYS:
        return True

    alias max_profiling_level = build_info_asyncrt_max_profiling_level()
    if not max_profiling_level:
        return False

    return level <= (
        (max_profiling_level.value() >> (type.value * kProfilingTypeWidthBits))
        & ((1 << kProfilingTypeWidthBits) - 1)
    )


@always_inline
fn is_profiling_disabled[type: TraceType, level: TraceLevel]() -> Bool:
    """Returns False if the profiling is enabled for that specific type and
    level and True otherwise."""
    return not is_profiling_enabled[type, level]()


@always_inline
fn is_mojo_profiling_enabled[level: TraceLevel]() -> Bool:
    """Returns whether Mojo profiling is enabled for the specified level."""
    return is_profiling_enabled[TraceType.MOJO, level]()


@always_inline
fn is_mojo_profiling_disabled[level: TraceLevel]() -> Bool:
    """Returns whether Mojo profiling is disabled for the specified level."""
    return is_profiling_disabled[TraceType.MOJO, level]()


@always_inline
fn trace_arg(name: String, shape: StaticIntTuple) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.
    """
    var s = name + "="
    for i in range(len(shape)):
        if i != 0:
            s += "x"
        s += str(shape[i])
    return s


@always_inline
fn trace_arg(name: String, shape: StaticIntTuple, dtype: DType) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.
    """
    return trace_arg(name, shape) + "x" + str(dtype)


@always_inline
fn trace_arg(name: String, buf: NDBuffer) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.
    """
    return trace_arg(name, buf.dynamic_shape, buf.type)


# ===----------------------------------------------------------------------===#
# Trace
# ===----------------------------------------------------------------------===#


@value
struct Trace[level: TraceLevel]:
    """An object representing a specific Mojo trace."""

    alias trace_type = TraceType.MOJO

    var name: StringLiteral
    var int_payload: Optional[Int]
    var detail: String
    var event_id: Int
    var parent_id: Int

    @always_inline
    fn __init__(
        inout self,
        name: StringLiteral,
        detail: String = "",
        parent_id: Int = 0,
    ):
        """Creates a Mojo trace with the given name.

        Args:
            name: The name that is used to identify this Mojo trace.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be
                appended to parent name. 0 (default) indicates no parent.
        """

        self.event_id = 0  # Known only when begin recording in __enter__
        self.parent_id = parent_id

        @parameter
        if is_profiling_disabled[Self.trace_type, level]():
            self.name = ""
            self.detail = ""
            self.int_payload = Optional[Int]()
        else:
            self.name = name
            self.detail = detail
            self.int_payload = Optional[Int]()

    @always_inline
    fn __init__(
        inout self,
        name: StringLiteral,
        task_id: Int,
        detail: String = "",
        parent_id: Int = 0,
    ):
        """Creates a Mojo trace with the given name.

        This does not start the trace range.

        Args:
            name: The name that is used to identify this Mojo trace.
            task_id: Int that is appended to name.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be appended to parent name.
        """

        self.event_id = 0  # Known only when begin recording in __enter__
        self.parent_id = parent_id

        @parameter
        if is_profiling_disabled[Self.trace_type, level]():
            self.name = ""
            self.detail = ""
            self.int_payload = Optional[Int]()
        else:
            self.name = name
            self.detail = detail
            self.int_payload = task_id

    @always_inline
    fn __enter__(inout self):
        """Enters the trace context.

        This begins recording of the trace event.
        """

        @parameter
        if is_profiling_disabled[Self.trace_type, level]():
            return

        if self.detail:
            # 1. If there is a detail string we must heap allocate the string
            #    because it presumably contains information only known at
            #    runtime.
            var detail_strref = self.detail._strref_dangerous()

            # Begins recording the trace range from the stack. This is only enabled if the AsyncRT
            # profiling is enabled.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBeginDetail", Int
            ](
                self.name.unsafe_cstr_ptr(),
                len(self.name),
                detail_strref.data,
                detail_strref.length.value,
                self.parent_id,
            )
        elif self.int_payload:
            # 2. If there is a task id, use the profiler API to create task:id
            #    labels without copying.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBeginTask", Int
            ](
                self.name.unsafe_cstr_ptr(),
                len(self.name),
                self.parent_id,
                self.int_payload.value(),
            )
        else:
            # 3. In the common case without a task id or detail string, create
            #    a profiler event without copying until explicit intern call.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBegin", Int
            ](self.name.unsafe_cstr_ptr(), len(self.name), self.parent_id)
        external_call[
            "KGEN_CompilerRT_TimeTraceProfilerSetCurrentId", NoneType
        ](self.event_id)

    @always_inline
    fn __exit__(self):
        """Exits the trace context.

        This finishes recording of the trace event.
        """

        @parameter
        if is_profiling_disabled[Self.trace_type, level]():
            return
        if self.event_id == 0:
            return
        external_call["KGEN_CompilerRT_TimeTraceProfilerEnd", NoneType](
            self.event_id
        )
        external_call[
            "KGEN_CompilerRT_TimeTraceProfilerSetCurrentId", NoneType
        ](0)

    @always_inline
    fn __exit__(self, e: Error) -> Bool:
        self.__exit__()
        return not e

    # WAR: passing detail_fn to __init__ causes internal compiler crash
    @staticmethod
    @always_inline
    fn _get_detail_str[detail_fn: fn () capturing -> String]() -> String:
        """Return the detail str when tracing is enabled and an empty string otherwise.
        """

        @parameter
        if is_profiling_enabled[Self.trace_type, level]():
            return detail_fn()
        else:
            return ""

    fn start(inout self):
        """Start recording trace event."""
        self.__enter__()

    fn end(inout self):
        """End recording trace event."""
        self.__exit__()


fn get_current_trace_id() -> Int:
    """Returns the id of last created trace entry on the current thread."""
    return external_call["KGEN_CompilerRT_TimeTraceProfilerGetCurrentId", Int]()
