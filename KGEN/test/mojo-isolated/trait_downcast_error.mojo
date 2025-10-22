# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# RUN: %parse-mojo-isolated -verify-diagnostics %s


struct Foo:
    pass


fn trait_downcast_concrete_type(x: Foo):
    # COM: mis-uses of downcast on a concrete type should be detected during
    # parsing time.

    # expected-error @below {{'Foo' value has no attribute 'copy'}}
    var _ = trait_downcast[Copyable](x).copy()
