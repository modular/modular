# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s


struct Spaceship:
    var location: Int

    fn set_location(mut self, new_location: Int):
        self.location = new_location


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    fn fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)


# // -----


struct Spaceship:
    var location: Int

    fn set_location(mut self, new_location: Int):
        self.location = new_location


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    fn fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)


# CHECK-LABEL: lit.fn @"do_things
fn do_things(mut ship: Spaceship):
    # CHECK: lit.call {{.*}}@"fly_to
    # CHECK-SAME: "self": !lit.ref<!Spaceship, mut *[0,0]>
    ship.fly_to(2)


# // -----

# Tests that we handle multiple extensions, and they're uniquely named.


struct Spaceship:
    var location: Int

    fn set_location(mut self, new_location: Int):
        self.location = new_location


# CHECK-LABEL: lit.struct.decl @Spaceship
# CHECK-NOT: lit.struct.decl @"extension:Spaceship"
# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship:
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    fn fly_to(mut self: Spaceship, new_location: Int):
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


struct Spaceship:
    var location: Int

    fn set_location(mut self, new_location: Int):
        self.location = new_location


trait Flying:
    fn fly_to(mut self, new_location: Int):
        ...


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: immediateParents = #M<symbols[@struct_extensions::@Flying]>
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship(Flying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    fn fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)


# CHECK: kgen.conformance @"struct_extensions::Flying" {
# CHECK-NEXT: kgen.witness "fly_to
# CHECK-SAME: = @struct_extensions::@"extension:Spaceship"::@"fly_to
# ConformanceOp's immediateParents should match the trait's immediateParents.
# CHECK-NEXT: } attributes {immediateParents = #M<symbols[@{{.*}}::@AnyType]>, traitRef = @struct_extensions::@Flying}


# // -----


trait Flying:
    fn fly_to(mut self, new_location: Int):
        ...


struct Spaceship:
    var location: Int

    fn set_location(mut self, new_location: Int):
        self.location = new_location


__extension Spaceship(Flying):
    fn fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)


fn launch_flying[F: Flying](mut flying: F):
    flying.fly_to(2)


# CHECK-LABEL: lit.fn @"launch_ship
fn launch_ship(mut ship: Spaceship):
    # CHECK: lit.call @struct_extensions::@"launch_flying[struct_extensions::Flying]
    # CHECK-SAME: <:!Flying !Spaceship>
    launch_flying(ship)


# // -----

# Define a capturing lambda type (like elementwise_epilogue_type)
comptime capturing_lambda_type = fn (Int) capturing -> Int


struct StructWithCapturingLambda[T: Int, my_lambda: capturing_lambda_type]:
    var value: Int

    # This part of the test is here to establish the fact that struct
    # methods get 'capturing' when the struct has a capturing fn parameter.
    # Further below, we'll CHECK that the same thing happens for extensions.
    # CHECK-LABEL: lit.fn @"helper_with_capturing_lambda
    # CHECK-SAME: capturing -> !Int
    fn helper_with_capturing_lambda(self) -> Int:
        return Self.T


__extension StructWithCapturingLambda:
    # CHECK-LABEL: lit.fn @"user_from_extension
    # This is the important check, that extension methods also get 'capturing'
    # if the struct has any capturing fn parameters.
    # CHECK-SAME: ) capturing -> !Int attributes
    fn user_from_extension(self) -> Int:
        return Self.helper_with_capturing_lambda(self)


# // -----


struct Spaceship:
    var location: Int

    fn set_location(mut self, new_location: Int):
        self.location = new_location


trait Flying:
    fn fly_to(mut self, new_location: Int):
        ...


# CHECK-LABEL: lit.trait.decl @Flying
# All traits implicitly inherit from AnyType (unless builtins are disabled)
# CHECK-SAME: immediateParents = #M<symbols[@{{.*}}::@AnyType]>


# CHECK-LABEL: lit.extension.decl @"extension:Spaceship"
# CHECK-SAME: immediateParents = #M<symbols[@struct_extensions::@Flying]>
# CHECK-SAME: targetStruct = @struct_extensions::@Spaceship
__extension Spaceship(Flying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    fn fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)


# CHECK: kgen.conformance @"struct_extensions::Flying" {
# CHECK-NEXT: kgen.witness "fly_to($0&,::Int)"
# CHECK-SAME: = @struct_extensions::@"extension:Spaceship"::@"fly_to
# ConformanceOp's immediateParents should match the trait's immediateParents.
# Since Flying inherits from AnyType, the conformance should have AnyType.
# CHECK-NEXT: } attributes {immediateParents = #M<symbols[@{{.*}}::@AnyType]>, traitRef = @struct_extensions::@Flying}

# // -----


struct ZDType:
    fn __init__(out self):
        pass


comptime ZScalar = ZSIMD[ZDType(), size=1]


struct ZSIMD[dtype: ZDType, size: Int]:
    pass


trait ZConvertibleToPython:
    pass


__extension ZSIMD(ZConvertibleToPython):
    pass


# // -----

# Tests accessing a struct's generic parameter from an extension.
# Makes sure that the `.d` correctly grabs the struct's alias, and not
# the one that's duplicated into the extension.
# TODO(MOCO-522): Arcana docs here!


struct Int:
    pass


struct MyContainer[d: Int]:
    pass


__extension MyContainer:
    pass


fn test_param_access[dtype: Int]():
    # Note the Int below, thats what makes sure it's working.
    # CHECK: lit.alias.decl *"element_type`": !Int = <dtype>
    comptime element_type = MyContainer[dtype].d


# // -----

# This test revealed a bug where struct signature resolution wasn't
# signature-resolving the target struct.

# The order here matters, extension must be first.
__extension MyThing:
    # In the extension's scope, `Self` didn't have generic parameters.
    @staticmethod
    fn foo() -> Self:
        return Self()


@fieldwise_init
struct MyThing[N: Int]:
    pass


fn zork(m: MyThing[5]):
    pass


fn bork(m: MyThing[5]):
    # This had a mismatch, because m.foo()'s return type was MyThing but
    # zork expected the proper MyThing[5].
    zork(m.foo())



# TODO(MOCO-522): Add tests for aliases in extensions
