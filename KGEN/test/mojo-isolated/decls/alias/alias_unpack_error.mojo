# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -import-mojo -verify-diagnostics -split-input-file %s

# expected-error @below {{cannot unpack value of 'Tuple[Int, FloatDyn]' of 2 elements into 3 values}}
alias a, (b, c, d) = (1, (2, 3.0))

# // -----

# expected-error @below {{invalid alias target: expected an identifier or '_'}}
alias t, True, c = 1, 2, 3


# // -----

struct A:
    # expected-error @below {{does not support alias destructuring in struct}}
    alias a, b = 1, 2


# // -----

trait A:
    # expected-error @below {{does not support alias destructuring in trait}}
    alias a, b = 1, 2


# // -----


# expected-note @below {{previous definition here}}
alias a, b = 1, 2
# expected-error @below {{invalid redefinition of 'b'}}
alias b, c = 2, 3
