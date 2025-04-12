# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides tracing utilities."""

from collections import Optional
from collections.optional import OptionalReg
from collections.string import StaticString
from sys import external_call
from sys.param_env import env_get_int, is_defined

import gpu.host._tracing as gpu_tracing
from buffer import NDBuffer
from gpu.host._tracing import _end_range as _end_gpu_range
from gpu.host._tracing import _is_enabled as _gpu_is_enabled
from gpu.host._tracing import _is_enabled_details as _gpu_is_enabled_details
from gpu.host._tracing import _mark as _mark_gpu
from gpu.host._tracing import _start_range as _start_gpu_range
from memory import UnsafePointer

from utils import IndexList, Variant


fn _build_info_asyncrt_max_profiling_level() -> OptionalReg[Int]:
    @parameter
    if not is_defined["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]():
        return None
    return env_get_int["MODULAR_ASYNCRT_MAX_PROFILING_LEVEL"]()


# ===-----------------------------------------------------------------------===#
# TraceCategory
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TraceCategory(EqualityComparable, Intable):
    """An enum-like struct specifying the type of tracing to perform."""

    alias OTHER = Self(0)
    alias ASYNCRT = Self(1)
    alias MEM = Self(2)
    alias Kernel = Self(3)
    alias MAX = Self(4)

    var value: Int

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: Self) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Self) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self != rhs

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return self.value


# ===-----------------------------------------------------------------------===#
# TraceLevel
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TraceLevel(EqualityComparable):
    """An enum-like struct specifying the level of tracing to perform."""

    alias ALWAYS = Self(0)
    alias OP = Self(1)
    alias THREAD = Self(2)

    var value: Int

    @always_inline
    @implicit
    fn __init__(out self, value: Int):
        self.value = value

    @always_inline("nodebug")
    fn __eq__(self, rhs: Self) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: Self) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __le__(self, rhs: Self) -> Bool:
        """Performs less than or equal to comparison.

        Args:
            rhs: The value to compare.

        Returns:
            True if this value is less than or equal to `rhs`.
        """
        return self.value <= rhs.value

    @always_inline("nodebug")
    fn __is__(self, rhs: Self) -> Bool:
        """Compares for equality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are equal.
        """
        return self == rhs

    @always_inline("nodebug")
    fn __isnot__(self, rhs: Self) -> Bool:
        """Compares for inequality.

        Args:
            rhs: The value to compare.

        Returns:
            True if they are not equal.
        """
        return self != rhs

    @always_inline("nodebug")
    fn __int__(self) -> Int:
        return self.value


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
fn is_profiling_enabled[type: TraceCategory, level: TraceLevel]() -> Bool:
    """Returns True if the profiling is enabled for that specific type and
    level and False otherwise."""
    alias kProfilingTypeWidthBits = 3

    @parameter
    if level == TraceLevel.ALWAYS:
        return True

    alias max_profiling_level = _build_info_asyncrt_max_profiling_level()
    if not max_profiling_level:
        return False

    return level <= (
        (max_profiling_level.value() >> (type.value * kProfilingTypeWidthBits))
        & ((1 << kProfilingTypeWidthBits) - 1)
    )


@always_inline
fn is_profiling_disabled[type: TraceCategory, level: TraceLevel]() -> Bool:
    """Returns False if the profiling is enabled for that specific type and
    level and True otherwise."""
    return not is_profiling_enabled[type, level]()


@always_inline
fn _is_gpu_profiler_enabled[type: TraceCategory, level: TraceLevel]() -> Bool:
    """Returns True if the e2e kernel profiling is enabled. Note that we always
    prefer to use llcl profiling if they are enabled."""
    return (
        is_profiling_disabled[type, level]()
        and level <= TraceLevel.OP
        and _gpu_is_enabled()
    )


@always_inline
fn _is_gpu_profiler_detailed_enabled[
    type: TraceCategory, level: TraceLevel
]() -> Bool:
    """Returns True if the e2e detailed kernel profiling is enabled. Note that
    we always prefer to use llcl profiling if they are enabled."""
    return (
        is_profiling_disabled[type, level]()
        and level <= TraceLevel.OP
        and _gpu_is_enabled_details()
    )


@always_inline
fn is_mojo_profiling_enabled[level: TraceLevel]() -> Bool:
    """Returns whether Mojo profiling is enabled for the specified level."""
    return is_profiling_enabled[TraceCategory.MAX, level]()


@always_inline
fn is_mojo_profiling_disabled[level: TraceLevel]() -> Bool:
    """Returns whether Mojo profiling is disabled for the specified level."""
    return is_profiling_disabled[TraceCategory.MAX, level]()


@always_inline
fn trace_arg(name: String, shape: IndexList) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.
    """
    var s = name + "="
    for i in range(len(shape)):
        if i != 0:
            s += "x"
        s += String(shape[i])
    return s


@always_inline
fn trace_arg(name: String, shape: IndexList, dtype: DType) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.
    """
    return trace_arg(name, shape) + "x" + String(dtype)


@always_inline
fn trace_arg(name: String, buf: NDBuffer) -> String:
    """Helper to stringify the type and shape of a kernel argument for tracing.
    """
    return trace_arg(name, buf.dynamic_shape, buf.type)


# ===-----------------------------------------------------------------------===#
# Trace
# ===-----------------------------------------------------------------------===#


@value
struct Trace[
    level: TraceLevel,
    *,
    category: TraceCategory = TraceCategory.MAX,
    target: Optional[StaticString] = None,
]:
    """An object representing a specific trace."""

    var _name_value: Variant[String, StaticString]
    var int_payload: OptionalReg[Int]
    var detail: String
    var event_id: Int
    var parent_id: Int

    # This constructor is intentionally hidden because Variant is too flexible
    # about what it allows and we want to ensure that only StaticString or
    # String are used.
    @always_inline
    fn __init__(
        out self,
        *,
        owned _name_value: Variant[String, StaticString],
        detail: String = "",
        parent_id: Int = 0,
        task_id: OptionalReg[Int] = None,
    ):
        """Creates a Mojo trace with the given name.

        Args:
            _name_value: The name that is used to identify this Mojo trace.
            detail: Details of the trace entry.
            parent_id: Parent to associate the trace with. Trace name will be
                appended to parent name. 0 (default) indicates no parent.
            task_id: Int that is appended to name.
        """

        self.event_id = 0  # Known only when begin recording in __enter__
        self.parent_id = parent_id

        # Debug assert the AsyncRT profiler => StaticString invariant for now,
        # to avoid making this raising.
        debug_assert(
            is_profiling_disabled[category, level]()
            or _name_value.isa[StaticString](),
            "the AsyncRT profiler only supports `StaticString` names",
        )

        @parameter
        if _is_gpu_profiler_enabled[category, level]():
            self._name_value = _name_value^

            @parameter
            if _gpu_is_enabled_details():
                self.detail = detail
            else:
                self.detail = ""
            self.int_payload = None
        elif is_profiling_enabled[category, level]():
            self._name_value = _name_value^
            self.detail = detail

            @parameter
            if target:
                if self.detail:
                    self.detail += ";"
                self.detail += "target=" + target.value()
            self.int_payload = task_id
        else:
            self._name_value = StaticString("")
            self.detail = ""
            self.int_payload = None

    @always_inline
    fn __init__(
        out self,
        owned name: String,
        detail: String = "",
        parent_id: Int = 0,
        *,
        task_id: OptionalReg[Int] = None,
    ):
        self = Self(
            _name_value=name^,
            detail=detail,
            parent_id=parent_id,
            task_id=task_id,
        )

    @always_inline
    fn __init__(
        out self,
        name: StaticString,
        detail: String = "",
        parent_id: Int = 0,
        *,
        task_id: OptionalReg[Int] = None,
    ):
        self = Self(
            _name_value=name,
            detail=detail,
            parent_id=parent_id,
            task_id=task_id,
        )

    @always_inline
    fn __init__(
        out self,
        name: StringLiteral,
        detail: String = "",
        parent_id: Int = 0,
        *,
        task_id: OptionalReg[Int] = None,
    ):
        self = Self(
            _name_value=StaticString(name),
            detail=detail,
            parent_id=parent_id,
            task_id=task_id,
        )

    @always_inline
    fn __enter__(mut self):
        """Enters the trace context.

        This begins recording of the trace event.
        """

        @parameter
        if _is_gpu_profiler_enabled[category, level]():

            @parameter
            if _gpu_is_enabled_details():
                # Convert to String since nvtx range APIs copy messages anyway.
                # TODO(KERN-1052): optimize by exposing explicit string
                # registration.
                self.event_id = Int(
                    _start_gpu_range(
                        message=self.name()
                        + (("/" + self.detail) if self.detail else ""),
                        category=Int(category),
                    )
                )
            else:
                self.event_id = Int(
                    _start_gpu_range(
                        message=self.name(), category=Int(category)
                    )
                )
            return

        @parameter
        if is_profiling_disabled[category, level]():
            return

        # The tracing builtins below expect the string to live beyond begin/end
        # calls, so we have to pass an inner pointer into the representation.
        #
        # IMPORTANT: since the AsyncRT profiler only supports `StaticString`
        # names, `self._name_value` must be `StaticString` when
        # `is_profiling_enabled()` is set.
        var name_str_ptr = self._name_value[StaticString].unsafe_ptr()
        var name_str_len = len(self._name_value[StaticString])

        if self.detail:
            # 1. If there is a detail string we must heap allocate the string
            #    because it presumably contains information only known at
            #    runtime.

            # Begins recording the trace range from the stack. This is only enabled if the AsyncRT
            # profiling is enabled.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBeginDetail", Int
            ](
                name_str_ptr,
                name_str_len,
                self.detail.unsafe_ptr(),
                len(self.detail),
                self.parent_id,
            )
        elif self.int_payload:
            # 2. If there is a task id, use the profiler API to create task:id
            #    labels without copying.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBeginTask", Int
            ](
                name_str_ptr,
                name_str_len,
                self.parent_id,
                self.int_payload.value(),
            )
        else:
            # 3. In the common case without a task id or detail string, create
            #    a profiler event without copying until explicit intern call.
            self.event_id = external_call[
                "KGEN_CompilerRT_TimeTraceProfilerBegin", Int
            ](
                name_str_ptr,
                name_str_len,
                self.parent_id,
            )

        external_call[
            "KGEN_CompilerRT_TimeTraceProfilerSetCurrentId", NoneType
        ](self.event_id)

    @always_inline
    fn __exit__(self):
        """Exits the trace context.

        This finishes recording of the trace event.
        """

        @parameter
        if _is_gpu_profiler_enabled[category, level]():
            _end_gpu_range(gpu_tracing.RangeID(self.event_id))
            return

        @parameter
        if is_profiling_disabled[category, level]():
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
    fn mark(self):
        """Marks the tracer with the info at the specific point of time."""

        @parameter
        if _is_gpu_profiler_enabled[category, level]():
            var message = self.name()

            @parameter
            if _gpu_is_enabled_details():
                if self.detail:
                    message += "/" + self.detail

            _mark_gpu(message=message)

    @always_inline
    fn name(self) -> String:
        return String(self._name_value[StaticString]) if self._name_value.isa[
            StaticString
        ]() else self._name_value[String]

    # WAR: passing detail_fn to __init__ causes internal compiler crash
    @staticmethod
    @always_inline
    fn _get_detail_str[detail_fn: fn () capturing -> String]() -> String:
        """Return the detail str when tracing is enabled and an empty string otherwise.
        """

        @parameter
        if (
            is_profiling_enabled[category, level]()
            or _is_gpu_profiler_detailed_enabled[category, level]()
        ):
            return detail_fn()
        else:
            return ""

    fn start(mut self):
        """Start recording trace event."""
        self.__enter__()

    fn end(mut self):
        """End recording trace event."""
        self.__exit__()


fn get_current_trace_id[level: TraceLevel]() -> Int:
    """Returns the id of last created trace entry on the current thread."""

    @parameter
    if is_mojo_profiling_enabled[level]():
        return external_call[
            "KGEN_CompilerRT_TimeTraceProfilerGetCurrentId", Int
        ]()
    else:
        return 0
