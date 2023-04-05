# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import assert_param_bool
from BuildInfo import build_info_llcl_max_profiling_level
from String import StringRef


# ===----------------------------------------------------------------------===#
# TraceCategory
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TraceType:
    alias OTHER = TraceType(0)
    alias LLCL = TraceType(1)
    alias MEM = TraceType(2)
    alias MOJO = TraceType(3)

    var value: Int

    @always_inline("nodebug")
    fn __init__(value: Int) -> TraceType:
        return TraceType {value: value}

    @always_inline("nodebug")
    fn __eq__(self, rhs: TraceType) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: TraceType) -> Bool:
        return self.value != rhs.value


# ===----------------------------------------------------------------------===#
# TraceLevel
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct TraceLevel:
    alias ALWAYS = TraceLevel(0)
    alias OP = TraceLevel(1)
    alias THREAD = TraceLevel(2)

    var value: Int

    @always_inline("nodebug")
    fn __init__(value: Int) -> Self:
        return Self {value: value}

    @always_inline("nodebug")
    fn __eq__(self, rhs: TraceLevel) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: TraceLevel) -> Bool:
        return self.value != rhs.value

    @always_inline("nodebug")
    fn __le__(self, rhs: TraceLevel) -> Bool:
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
    return is_profiling_enabled[TraceType.MOJO, level]()


@always_inline
fn is_mojo_profiling_disabled[level: TraceLevel]() -> Bool:
    return is_profiling_disabled[TraceType.MOJO, level]()


# ===----------------------------------------------------------------------===#
# trace_range_push
# ===----------------------------------------------------------------------===#


@always_inline
fn trace_range_push[
    type: TraceType, level: TraceLevel
](name: StringRef, detail: StringRef):
    """Push a trace event onto a per-thread stack of traces.
    Should be paired with calls to trace_range_pop().

    The modular stack needs to be configured with MODULAR_LLCL_MAX_PROFILING_LEVEL
    specifing the profiling level to profile at.

    The trace events will be written to the profiling file passed to the
    LLCL Runtime ctor.

    Args:
        name: name of the trace event (will be copied).
        detail: additional details about the trace event (will be copied).

    """

    @parameter
    if is_profiling_disabled[type, level]():
        return

    # Pushe the trace range from the stack. This is only enabled if the LLCL
    # profiling is enabled.
    __mlir_op.`pop.external_call`[
        func : __mlir_attr.`@KGEN_CompilerRT_TimeTraceProfilerBegin`,
        _type:[],
    ](
        name.data.address,
        name.length.value,
        detail.data.address,
        detail.length.value,
    )


# ===----------------------------------------------------------------------===#
# trace_range_pop
# ===----------------------------------------------------------------------===#


@always_inline
fn trace_range_pop[type: TraceType, level: TraceLevel]():
    """Pop a trace event off a per-thread stack of traces.
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
        func : __mlir_attr.`@KGEN_CompilerRT_TimeTraceProfilerEnd`,
        _type:[],
    ]()


# ===----------------------------------------------------------------------===#
# Trace
# ===----------------------------------------------------------------------===#


struct Trace[level: TraceLevel]:
    alias trace_type = TraceType.MOJO

    fn __copy__(self) -> Self:
        return Self {}

    fn __init__(name: StringRef) -> Trace[level]:
        return Trace[level](name, "")

    fn __init__(name: StringRef, detail: StringRef) -> Trace[level]:
        trace_range_push[trace_type, level](name, detail)
        return Trace[level] {}

    fn __del__(self&):
        trace_range_pop[trace_type, level]()
