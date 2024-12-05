# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes a simple GPU profiler."""

from builtin._location import __call_location, _SourceLocation
from builtin.io import _printf

from time import perf_counter_ns


@value
struct ProfileBlock[enabled: Bool = False]:
    var name: StringLiteral
    var loc: _SourceLocation
    var start_time: UInt

    @always_inline
    @implicit
    fn __init__(out self, name: StringLiteral):
        self.start_time = 0

        @parameter
        if enabled:
            self.name = name
            self.loc = __call_location()
        else:
            self.name = ""
            self.loc = _SourceLocation(0, 0, "")

    @always_inline
    fn __enter__(mut self):
        @parameter
        if not enabled:
            return
        self.start_time = perf_counter_ns()

    @always_inline
    fn __exit__(mut self):
        @parameter
        if not enabled:
            return

        var end_time = perf_counter_ns()

        _printf["@ %s %ld\n"](
            self.name.unsafe_cstr_ptr(), self.start_time - end_time
        )
