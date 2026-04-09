# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# Module imported by deprecated_errors.mojo to test cross-module deprecation warnings.


@deprecated("use of deprecated struct 'DeprecatedInAnotherModule'")
# expected-note @below {{'DeprecatedInAnotherModule' declared here}}
struct DeprecatedInAnotherModule:
    pass
