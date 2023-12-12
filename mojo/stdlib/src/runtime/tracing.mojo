# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides tracing utilities."""

from sys._build import build_info_llcl_max_profiling_level
from utils._optional import Optional

# ===----------------------------------------------------------------------===#
# TraceCategory
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct TraceType:
    """An enum-like struct specifying the type of tracing to perform."""

    alias OTHER = TraceType(0)
    alias LLCL = TraceType(1)
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
struct TraceLevel:
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
    alias MODULAR_LLCL_MAX_PROFILING_LEVEL = (
        build_info_llcl_max_profiling_level()
    )
    return level <= (
        (
            MODULAR_LLCL_MAX_PROFILING_LEVEL
            >> (type.value * kProfilingTypeWidthBits)
        )
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
    fn __enter__(self):
        """Enters the trace context.

        This pushes this trace event onto a per-thread stack of traces and starts
        the trace range.

        Note: Traces cannot currently cross async/await boundaries since they use the
        implicit per-thread profiling stack.
        """

        @parameter
        if is_profiling_disabled[Self.trace_type, level]():
            return

        let name = self.__str__()
        let name_strref = name._strref_dangerous()
        let detail_strref = self.detail._strref_dangerous()

        # Pushes the trace range from the stack. This is only enabled if the LLCL
        # profiling is enabled.
        __mlir_op.`pop.external_call`[
            func = "KGEN_CompilerRT_TimeTraceProfilerBegin".value,
            _type=None,
        ](
            name_strref.data,
            name_strref.length.value,
            detail_strref.data,
            detail_strref.length.value,
            self.parent_id,
        )
        name._strref_keepalive()

    @always_inline
    fn __exit__(self):
        """Exits the trace context.

        This pops this trace event off a per-thread stack of traces and stops the
        trace range.

        Note: Traces cannot currently cross async/await boundaries since they use the
        implicit per-thread profiling stack.
        """

        @parameter
        if is_profiling_disabled[Self.trace_type, level]():
            return
        # Pop the trace range from the stack. This is only enabled if the LLCL
        # profiling is enabled.
        __mlir_op.`pop.external_call`[
            func = "KGEN_CompilerRT_TimeTraceProfilerEnd".value,
            _type=None,
        ]()

    @always_inline
    fn __str__(self) -> String:
        constrained[
            is_profiling_enabled[Self.trace_type, level](),
            "cannot get trace string in non-profiling build",
        ]()
        let name = self.name + String(
            self.int_payload.value()
        ) if self.int_payload else self.name
        return name

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


fn get_current_trace_id() -> Int:
    """Get id of trace currently on the top of the per-thread stack of traces.
    """
    return __mlir_op.`pop.external_call`[
        func = "KGEN_CompilerRT_TimeTraceProfilerCurrentId".value,
        _type=Int,
    ]()
