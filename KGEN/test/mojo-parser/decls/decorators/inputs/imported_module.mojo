# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct DTypePointer(Movable where False):
    pass


@deprecated("use of deprecated struct 'DeprecatedInAnotherModule'")
# expected-note @below {{'DeprecatedInAnotherModule' declared here}}
struct DeprecatedInAnotherModule(Movable where False):
    pass
