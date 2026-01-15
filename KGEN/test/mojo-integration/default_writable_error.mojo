# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo -D test=1 %s 2>&1 | FileCheck --check-prefix CHECK_1 %s
# RUN: not %mojo -D test=2 %s 2>&1 | FileCheck --check-prefix CHECK_2 %s

# Test that the default Writable implementation produces a clear error message
# when a field does not implement Writable.

from sys import env_get_int


@fieldwise_init
struct NotWritable(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct HasBadField(Writable):
    var field: NotWritable


fn main():
    var value = HasBadField(NotWritable(1))
    var string = String()

    @parameter
    if env_get_int["test"]() == 1:
        # CHECK_1: constraint failed: Could not derive Writable for HasBadField
        value.write_to(string)
    elif env_get_int["test"]() == 2:
        # CHECK_2: constraint failed: Could not derive Writable for HasBadField
        value.write_repr_to(string)
