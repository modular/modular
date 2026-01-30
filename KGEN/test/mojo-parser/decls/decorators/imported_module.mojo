# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct DTypePointer:
    pass


@deprecated("use of deprecated struct 'DeprecatedInAnotherModule'")
# expected-note @below {{'DeprecatedInAnotherModule' declared here}}
struct DeprecatedInAnotherModule:
    pass
