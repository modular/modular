# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that three truly adjacent breakpoint() calls (no statements between)
# each produce a separate stop and can each be resumed past.


def main():
    breakpoint()
    breakpoint()
    breakpoint()
    print("done")
