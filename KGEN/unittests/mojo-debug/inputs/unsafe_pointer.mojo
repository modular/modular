# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# NOTE: A parallel copy of this file lives at
#   KGEN/test/mojo-debug/Inputs/unsafe_pointer.mojo
# The two files must be kept in sync. Any change here (adding variables,
# reordering lines) requires updating the sibling file AND updating the
# hardcoded line numbers in StdlibTypesTest.cpp / unsafe-pointer-formatter.lldb.


def keep_alive[*Ts: AnyType](*args: *Ts):
    pass


def main():
    var p_int = alloc[Int]({count = 1}).unsafe_leak()
    p_int[0] = 42
    keep_alive(p_int)  # breakpoint

    var p_neg = alloc[Int]({count = 1}).unsafe_leak()
    p_neg[0] = -5
    keep_alive(p_neg)  # breakpoint

    var p_bool = alloc[Bool]({count = 1}).unsafe_leak()
    p_bool[0] = True
    keep_alive(p_bool)  # breakpoint

    var p_float = alloc[Float64]({count = 1}).unsafe_leak()
    p_float[0] = Float64(3.125)
    keep_alive(p_float)  # breakpoint
