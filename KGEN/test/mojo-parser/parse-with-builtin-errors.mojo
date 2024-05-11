# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s


# expected-note @below {{'Error' declared here}}
fn Error():
    pass


struct FailingStruct:
    # expected-error @below {{'Error' doesn't resolve to a type}}
    fn failure(owned self) raises:
        pass
