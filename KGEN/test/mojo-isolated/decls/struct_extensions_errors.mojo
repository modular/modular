# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s

# @expected-note @below {{conflicts with this previous declaration}}
fn Spaceship():
    pass

# @expected-error @below {{cannot define a struct here with name 'Spaceship'}}
struct Spaceship:
    pass

# // -----


# @expected-note @below {{conflicts with this previous declaration}}
fn Spaceship():
    pass

# @expected-error @below {{cannot define an extension here with name 'Spaceship'}}
__extension Spaceship:
    pass
