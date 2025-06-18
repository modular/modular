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
    alias list = [1, 2, 3]
    cnt = 0

    @parameter
    for i in list:
        cnt += i
    debug_assert(cnt == 6)


fn main():
    test_unroll_warn_threshold()
    test_for_list()
