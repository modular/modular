# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s --verify-diagnostics

# Test that the default Equatable implementation produces a clear error message
# when a field does not implement Equatable.


@fieldwise_init
struct NotEquatable(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct HasBadField(Equatable):
    var field: NotEquatable


# expected-note @below {{constraint failed: Could not derive Equatable for {{.*}}HasBadField - member field `field: {{.*}}NotEquatable` does not implement Equatable}}
fn main():
    var a = HasBadField(NotEquatable(1))
    var b = HasBadField(NotEquatable(1))
    # expected-error @below {{call expansion failed}}
    _ = a == b
