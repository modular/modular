# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: not %mojo %s --verify-diagnostics

# Test that the default Equatable implementation produces a clear error message
# when a field does not implement Equatable.


@fieldwise_init
struct NotHashable(ImplicitlyCopyable):
    var x: Int


@fieldwise_init
struct HasBadField(Hashable):
    var field: NotHashable


# expected-note @below {{constraint failed: Could not derive Hashable for {{.*}}HasBadField - member field `field: {{.*}}NotHashable` does not implement Hashable}}
fn main():
    var a = HasBadField(NotHashable(1))
    # expected-error @below {{call expansion failed}}
    hash(a)
