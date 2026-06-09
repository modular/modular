# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


# Module imported by unavailable_errors.mojo to test cross-module
# unavailability errors.


@unavailable("use of unavailable function in another module")
# expected-note @below {{'unavailable_in_another_module' declared here}}
def unavailable_in_another_module():
    ...
