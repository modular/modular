# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics -split-input-file %s

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

# Ambiguous Lookup Case for Two Structs Via One Extension  (ALCFTSVOE):
# This is a case where we accidentally reference multiple structs, and one of
# them is accessed via an extension.


# @expected-note @below {{conflicts with this previous struct declaration}}
struct Spaceship:
    var fuel: Int


# @expected-error @below {{invalid redefinition of 'Spaceship'}}
struct Spaceship:
    pass


__extension Spaceship:
    pass


fn foo(ship: Spaceship) -> Int:  # shouldn't crash here
    return ship.fuel  # shouldn't crash here either


# // -----


# Tests we can call a constructor defined in an extension.


# CHECK-LABEL: lit.struct.decl @PlainStruct
struct PlainStruct:
    pass


# CHECK-LABEL: lit.extension.decl @"extension:PlainStruct"
# CHECK-SAME: targetStruct = @struct_extensions::@PlainStruct
__extension PlainStruct:
    # CHECK-LABEL: lit.fn @"__init__
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"zork
fn zork():
    # CHECK: lit.call {{.*}}@"__init__
    var z = PlainStruct()


# // -----


# Test when an extension method collides with a variable in the struct.


struct BaseStruct:
    # expected-note @below {{extension method conflicts with struct declaration}}
    var colliding: Int


__extension BaseStruct:
    # expected-error @below {{invalid redefinition of 'colliding'}}
    fn colliding(self) -> Int:
        return self.colliding


fn test_collisions(s: BaseStruct):
    var result = s.colliding()


# // -----


# Test when an extension method collides with an alias in the struct.


struct BaseStruct:
    # expected-note @below {{extension method conflicts with struct declaration}}
    comptime colliding: Int = 42


__extension BaseStruct:
    # expected-error @below {{invalid redefinition of 'colliding'}}
    fn colliding(self) -> Int:
        return self.colliding


fn test_collisions(s: BaseStruct):
    var result = s.colliding()


# // -----


# Test when an extension alias collides with an alias in the struct.


struct BaseStruct:
    # expected-note @below {{extension declaration conflicts with struct declaration}}
    comptime colliding: Int = 42


__extension BaseStruct:
    # expected-error @below {{invalid redefinition of 'colliding'}}
    comptime colliding: Int = 43


fn test_collisions(s: BaseStruct):
    var result = s.colliding
