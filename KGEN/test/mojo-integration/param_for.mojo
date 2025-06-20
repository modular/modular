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


fn main():
    test_unroll_warn_threshold()
    test_for_list()
