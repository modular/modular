# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that three breakpoint() calls with statements between them each
# produce a separate stop and can each be resumed past.


def main():
    var x = 0
    # fmt: off
    breakpoint(); x += 1  # stop_1
    breakpoint(); x += 1  # stop_2
    breakpoint(); x += 1  # stop_3
    # fmt: on
    print(x)
