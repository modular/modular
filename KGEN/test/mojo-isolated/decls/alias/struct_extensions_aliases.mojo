# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file %s | FileCheck %s


struct Spaceship:
    var fuel: Int


__extension Spaceship:
    comptime MaxSpeed: Int = 42


# CHECK-LABEL: lit.fn @"test_function
fn test_function():
    # CHECK: lit.alias.decl *"MySpeed`"
    comptime MySpeed = Spaceship.MaxSpeed
    # CHECK: lit.var.decl "speed"
    var speed: Int = MySpeed


# // -----

# Test accessing an alias whose type is another alias, and materializing it.


struct ZInt(ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __init__(out self, other: ZInt):
        pass


struct Spaceship:
    var fuel: ZInt


__extension Spaceship:
    comptime InnerType: AnyType = ZInt
    comptime MaxSpeed: InnerType = ZInt()


# CHECK-LABEL: lit.fn @"test_function()"
fn test_function():
    # Note how this is resolving to ZInt right here, it means the lookup worked.
    # CHECK: lit.alias.decl *"MySpeed`": !alias_InnerType1 = <sugar_member_alias(!Spaceship, "MaxSpeed",
    comptime MySpeed = Spaceship.MaxSpeed
    # Note how this is resolving to ZInt right here, it means the lookup worked.
    # CHECK: lit.var.decl "speed" var : !lit.ref<:!AnyType #alias_InnerType,
    var speed = MySpeed


# // -----

# Test an extension method accessing and materializing an extension alias.


struct ZInt(ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __init__(out self, other: ZInt):
        pass


struct Spaceship:
    var fuel: ZInt


__extension Spaceship:
    comptime MaxSpeed = ZInt()

    # CHECK-LABEL: lit.fn @"get_max_speed
    fn get_max_speed(self: Spaceship) -> ZInt:
        # Note how it's ZInt right here, it means the lookup worked.
        # CHECK: kgen.param.materialize: !ZInt
        return ZInt(MaxSpeed)


# // -----

# Test an extension method accessing a struct alias via self argument.


struct ZInt(ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __init__(out self, other: ZInt):
        pass


struct Rocket:
    comptime DefaultFuel = ZInt()
    var fuel: ZInt


__extension Rocket:
    # CHECK-LABEL: lit.fn @"get_default_fuel
    fn get_default_fuel(self) -> ZInt:
        # Note how it's ZInt right here, it means the lookup worked.
        # CHECK: kgen.param.materialize: !ZInt
        return self.DefaultFuel  # access via self


## // -----

# Tests an extension's alias accessing its struct's parameter declaration.


struct Rocket[T: AnyType]:
    pass


__extension Rocket:
    comptime FuelType = Self.T


# // -----

# Test accessing a generic struct's extension's alias
# Also tests calling an extension method on a generic container


struct ZInt(ImplicitlyCopyable):
    fn __init__(out self):
        pass

    fn __init__(out self, other: ZInt):
        pass


struct Container[T: ImplicitlyCopyable]:
    var data: Self.T


__extension Container:
    comptime ElementType = Self.T
    comptime DefaultSize = ZInt()

    fn get_element_via_self(self: Container[Self.T]) -> Self.T:
        return self.data


# CHECK-LABEL: lit.fn @"test_self_alias_with_generic_1
fn test_self_alias_with_generic_1(container: Container[ZInt]):
    # Note how it's ZInt right here, it means the lookup worked.
    # CHECK: lit.alias.decl *"MyElementType`1": !ImplicitlyCopyable =
    # CHECK-SAME: <sugar_member_alias(!lit.struct<#Container <:!ImplicitlyCopyable !ZInt>>, "ElementType", !ZInt)>
    comptime MyElementType = Container[ZInt].ElementType


# CHECK-LABEL: lit.fn @"test_self_alias_with_generic_2
fn test_self_alias_with_generic_2(container: Container[ZInt]):
    # Note how it's ZInt right here, it means the lookup worked.
    # CHECK: %element = lit.var.decl "element" var : !lit.ref<!ZInt,
    var element = container.get_element_via_self()
