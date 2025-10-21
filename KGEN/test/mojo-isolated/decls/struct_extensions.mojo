# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --mojo-disable-builtins -split-input-file | FileCheck %s

alias int = __mlir_type.index

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)


# // -----


alias int = __mlir_type.index
alias `2` = __mlir_attr.`2 : index`

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)

# CHECK-LABEL: lit.fn @"do_things
fn do_things(mut ship: Spaceship):
    # CHECK: lit.call {{.*}}@"fly_to
    # CHECK-SAME: "self": !lit.ref<!Spaceship, mut *[0,0]>
    ship.fly_to(`2`)

# // -----

# Tests that we handle multiple extensions, and they're uniquely named.

alias int = __mlir_type.index

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

# CHECK-LABEL: lit.struct.decl @Spaceship
# CHECK-NOT: lit.struct.decl @"extension:Spaceship"
# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship
# CHECK-NOT: lit.extension.decl @"extension:Spaceship"
# Note the quote at the end of this last line -------^
# This checks that we're naming this struct extension something different.
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    fn something_else(self: Spaceship):
        pass

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

# Test we can overload between struct and extension..

struct BaseStruct:
    fn same_name(self):
        pass

__extension BaseStruct:
    fn same_name(self, i: __mlir_type.index):
        pass

fn test_overloads(s: BaseStruct):
    var result = s.same_name()


# // -----

# Tests a struct extension with a trait

alias int = __mlir_type.index
alias `2` = __mlir_attr.`2 : index`


struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

trait Flying:
    fn fly_to(mut self, new_location: int):
        ...

# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: immediateParents = #M<symbols[@struct_extensions::@Flying]>
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship(Flying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: index
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)

# CHECK: kgen.conformance @"struct_extensions::Flying" {
# CHECK-NEXT: kgen.witness "fly_to{{.*}}" : {{.*}} = @struct_extensions::@"extension:Spaceship"::@"fly_to{{.*}}"
# CHECK-NEXT: } attributes {traitRef = @struct_extensions::@Flying}
# Flying trait has no parents, so ConformanceOp should not have immediateParents
# CHECK-NOT: } attributes {{{.*}}immediateParents


# // -----


alias int = __mlir_type.index
alias `2` = __mlir_attr.`2 : index`

trait Flying:
    fn fly_to(mut self, new_location: int):
        ...

struct Spaceship:
    var location: int
    fn set_location(mut self, new_location: int):
        self.location = new_location

__extension Spaceship(Flying):
    fn fly_to(mut self: Spaceship, new_location: int):
        self.set_location(new_location)


fn launch_flying[F: Flying](mut flying: F):
    flying.fly_to(`2`)

# CHECK-LABEL: lit.fn @"launch_ship
fn launch_ship(mut ship: Spaceship):
    # CHECK: lit.call @struct_extensions::@"launch_flying[struct_extensions::Flying]
    # CHECK-SAME: <:!Flying !Spaceship>
    launch_flying(ship)


# TODO(MOCO-522): Add tests for aliases in extensions
