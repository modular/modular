# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -verify-diagnostics -split-input-file %s

trait AnyType:
    pass

# @expected-note @below {{extension already assumes these parameter declarations}}
struct Spaceship[T: AnyType]:
    pass

# @expected-error @below {{cannot specify parameter declarations on extensions}}
__extension Spaceship[T: AnyType]:
    pass

# // -----

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

# @expected-error @below {{can't find a struct named 'Spaceship'}}
# @expected-error @below {{cannot define an extension here with name 'Spaceship'}}
__extension Spaceship:
    pass

# // -----

# @expected-error @below {{can't find a struct named 'Spaceship'}}
__extension Spaceship:
    pass
