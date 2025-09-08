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

# // -----

alias int = __mlir_type.index

# Ambiguous Lookup Case for Two Structs Via One Extension  (ALCFTSVOE):
# This is a case where we accidentally reference multiple structs, and one of
# them is accessed via an extension.


# @expected-note @below {{conflicts with this previous struct declaration}}
struct Spaceship:
    var fuel: int


# @expected-error @below {{invalid redefinition of 'Spaceship'}}
struct Spaceship:
    pass

__extension Spaceship:
    pass


fn foo(ship: Spaceship) -> int: # shouldn't crash here
    return ship.fuel # shouldn't crash here either
