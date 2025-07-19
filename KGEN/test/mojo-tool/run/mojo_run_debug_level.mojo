# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that all debug level options are accepted by the compiler.
# This test only validates that the options are parsed correctly.
# See mojo-tool/build/mojo_build_debug_level.mojo for tests that
# verify debug info is actually generated/omitted.
# RUN: %mojo --debug-level=none %s
# RUN: %mojo -g0 %s
# RUN: %mojo --debug-level=line-tables %s
# RUN: %mojo -g1 %s
# RUN: %mojo --debug-level=full %s
# RUN: %mojo -g %s
# RUN: %mojo -g2 %s


# CHECK: debug test ok
fn main() -> None:
    print("debug test ok")
