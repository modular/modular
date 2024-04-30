# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    var sum = 0
    for i in range(0, 10):
        sum += i
        if i == 8:
            breakpoint()
            # FIXME(33005): We should be able to test resuming after hitting the breakpoint,
            # but that is broken on Graviton.
            print(i)
