# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: mojo build --debug-level full -O0 %s -o %t
# RUN: mojo build --debug-level line-tables -O0 %s -o %t

# RUN: mojo build --debug-level full %s -o %t
# RUN: mojo build --debug-level line-tables %s -o %t


fn main():
    print("success")
