# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo --loop-unrolling-warn-threshold=27  %s --verify-diagnostics


fn main():
    var cnt = 0

    @parameter
    # expected-warning @+1 {{parameter for unrolling loop more than 27 times may cause long compilation time and large code size}}
    for i in range(28):
        cnt += 1
    print(cnt)
