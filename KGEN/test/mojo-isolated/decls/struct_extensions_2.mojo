# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s -split-input-file | FileCheck %s

# TODO(MOCO-522): This is a temporary test, this is only here because this test
# is different from the other struct extensions tests in that it doesnt use
# --mojo-disable-builtins. --mojo-disable-builtins makes it so structs',
# extensions' and conformances' immediateParents doesn't contain AnyType.


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
# CHECK-SAME: immediateParents = #M<symbols[@struct_extensions_2::@Flying]>
# CHECK-SAME: targetStruct = @struct_extensions_2::@Spaceship
__extension Spaceship(Flying):
    # CHECK-LABEL: lit.fn @"fly_to
    # CHECK-SAME: %self: !lit.ref<!Spaceship, mut *"{{.*}}">
    # CHECK-SAME: %new_location: !Int
    fn fly_to(mut self: Spaceship, new_location: Int):
        self.set_location(new_location)


# CHECK: kgen.conformance @"struct_extensions_2::Flying" {
# CHECK-NEXT: kgen.witness "fly_to{{.*}}" : {{.*}} = @struct_extensions_2::@"extension:Spaceship"::@"fly_to{{.*}}"
# ConformanceOp's immediateParents should match the trait's immediateParents.
# Since Flying inherits from AnyType, the conformance should have AnyType.
# CHECK-NEXT: } attributes {immediateParents = #M<symbols[@{{.*}}::@AnyType]>, traitRef = @struct_extensions_2::@Flying}

# // -----


struct ZDType:
    fn __init__(out self):
        pass


alias ZScalar = ZSIMD[ZDType(), size=1]


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
    alias element_type = MyContainer[dtype].d
