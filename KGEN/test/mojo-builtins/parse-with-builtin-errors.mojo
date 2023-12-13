# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s


# expected-note @below {{'Error' identifier redeclared here}}
fn Error():
    pass


struct FailingStruct:
    # expected-error @below {{builtin 'Error' identifier does not denote a type}}
    fn failure(owned self) raises:
        pass
