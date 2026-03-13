# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# We can run this file with optimizations disabled.
# RUN: %mojo --no-optimization %s
# RUN: %mojo -O0 %s


# CHECK: ok
def main() -> None:
    print("ok")
