# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# Test that 'try' statement in type bodies emits an error.
# This is in a separate file because after 'try' is rejected, the 'except'
# keyword becomes an invalid token, causing a cascading error.

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# try in struct body
##===----------------------------------------------------------------------===##

struct StructWithTry:
    var x: Int
    # expected-error @below {{'try' must be contained in a function}}
    try:
        pass
    # expected-error @below {{unexpected token in expression}}
    except:
        pass

##===----------------------------------------------------------------------===##
# try in trait body
##===----------------------------------------------------------------------===##

trait TraitWithTry:
    # expected-error @below {{'try' must be contained in a function}}
    try:
        pass
    # expected-error @below {{unexpected token in expression}}
    except:
        pass

##===----------------------------------------------------------------------===##
# try in extension body
##===----------------------------------------------------------------------===##

struct ExtendedStructWithTry:
    var x: Int

__extension ExtendedStructWithTry:
    # expected-error @below {{'try' must be contained in a function}}
    try:
        pass
    # expected-error @below {{unexpected token in expression}}
    except:
        pass
