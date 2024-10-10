# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Raise
##===----------------------------------------------------------------------===##


def raisingFunction():
    pass


# expected-note @below {{or mark surrounding function as 'raises'}}
fn callRaisingFunction():
    # expected-error @below {{cannot call function that may raise in a context that cannot raise}}
    # expected-note @below {{try surrounding the call in a 'try' block}}
    raisingFunction()


fn cannotReRaise() raises:
    # expected-error @below {{no contextual error to reraise}}
    # expected-note @below {{provide an error to raise or place 'raise' statement inside an except region}}
    raise


# expected-note @below {{or mark surrounding function as 'raises'}}
fn cannotRaise(err: Error):
    # expected-error @below {{cannot raise error in this context}}
    # expected-note @below {{try surrounding 'raise' in a 'try' block}}
    raise err


# Issue #12358
fn raise_bad_type() raises:
    raise 42  # expected-error {{cannot implicitly convert 'IntLiteral' value to 'Error'}}
