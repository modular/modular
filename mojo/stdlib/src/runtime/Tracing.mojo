# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Provides tracing utilities."""

from BuildInfo import build_info_llcl_max_profiling_level


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
# trace_range_push
# ===----------------------------------------------------------------------===#


@always_inline
fn trace_range_push[
    type: TraceType, level: TraceLevel
](name: StringRef, detail: StringRef):
    """Pushes a trace event onto a per-thread stack of traces.
    Should be paired with calls to trace_range_pop().

    The modular stack needs to be configured with MODULAR_LLCL_MAX_PROFILING_LEVEL
    specifying the profiling level to profile at.

    The trace events will be written to the profiling file passed to the
    LLCL Runtime ctor.

    Args:
        name: Name of the trace event (will be copied).
        detail: Additional details about the trace event (will be copied).

    """

    @parameter
    if is_profiling_disabled[type, level]():
        return

    # Pushes the trace range from the stack. This is only enabled if the LLCL
    # profiling is enabled.
    __mlir_op.`pop.external_call`[
        func : "KGEN_CompilerRT_TimeTraceProfilerBegin".value,
        _type:None,
    ](
        name.data,
        name.length.value,
        detail.data,
        detail.length.value,
    )


# ===----------------------------------------------------------------------===#
# trace_range_pop
# ===----------------------------------------------------------------------===#


@always_inline
fn trace_range_pop[type: TraceType, level: TraceLevel]():
    """Pops a trace event off a per-thread stack of traces.
    Should be paired with calls to trace_range_push().

    PROFILING_ON must be set to True otherwise this is a noop that will be
    folded away.

    The trace events will be written to the profiling file passed to the
    LLCL Runtime ctor.
    """

    @parameter
    if is_profiling_disabled[type, level]():
        return
    # Pop the trace range from the stack. This is only enabled if the LLCL
    # profiling is enabled.
    __mlir_op.`pop.external_call`[
        func : "KGEN_CompilerRT_TimeTraceProfilerEnd".value,
        _type:None,
    ]()


# ===----------------------------------------------------------------------===#
# Trace
# ===----------------------------------------------------------------------===#


@value
struct Trace[level: TraceLevel]:
    """An object representing a specific Mojo trace."""

    alias trace_type = TraceType.MOJO

    var name: StringRef
    var detail: StringRef

    fn __init__(inout self, name: StringRef):
        """Creates a Mojo trace with the given name.

        Args:
            name: The name that is used to identify this Mojo trace.
        """
        self = Self(name, "")

    fn __enter__(self):
        """Enters the trace context.

        This pushes this trace event onto a per-thread stack of traces.
        """
        trace_range_push[Self.trace_type, level](self.name, self.detail)

    fn __exit__(self):
        """Exits the trace context.

        This pops this trace event off a per-thread stack of traces.
        """
        trace_range_pop[Self.trace_type, level]()
