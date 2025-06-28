# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo --loop-unrolling-warn-threshold=27  %s --verify-diagnostics


fn test_unroll_warn_threshold():
    var cnt = 0

    @parameter
    # expected-warning @+1 {{parameter for unrolling loop more than 27 times may cause long compilation time and large code size}}
    for i in range(28):
        cnt += 1
    debug_assert(cnt == 28)


fn test_for_list():
    cnt = 0

    @parameter
    for i in [1, 2, 3]:
        cnt += i
    debug_assert(cnt == 6)

    # Test for floating point numbers.
    fp_cnt = 0.0

    @parameter
    for i in [1.0, 2.0, 3.0]:
        fp_cnt += i
    debug_assert(fp_cnt == 6.0)

    # Test for strings
    concated = String("")

    @parameter
    for str in [String("a"), "b", "c"]:
        var str2 = str  # Work around origin issue.
        concated += str2
    debug_assert(concated == "abc")


# derived from https://github.com/modular/modular/issues/4566
fn test_critical_edge():
    var a = 0

    @parameter
    for i in range(10):
        a = i  # Compiler hung here
    debug_assert(a == 9)


# derived from https://github.com/modular/modular/issues/4836
fn test_else_block():
    var a: Int  # Init not required because always assigned in else.

    @parameter
    for i in range(10):
        pass
    else:
        a = 1  # This should execute.
    debug_assert(a == 1)

    @parameter
    for i in range(10):
        if i == 4:
            break
    else:
        debug_assert(False)  # This should NOT execute.


fn main():
    test_unroll_warn_threshold()
    test_for_list()
    test_critical_edge()
    test_else_block()
