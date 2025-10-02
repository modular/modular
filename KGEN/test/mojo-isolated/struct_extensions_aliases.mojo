# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated --mojo-disable-builtins -split-input-file %s | FileCheck %s

alias int = __mlir_type.index
alias `42` = __mlir_attr.`42 : index`

struct Spaceship:
    var fuel: int

__extension Spaceship:
    alias MaxSpeed: int = `42`

# CHECK-LABEL: lit.fn @"test_function
fn test_function():
    # CHECK: lit.alias.decl *"MySpeed`"
    alias MySpeed = Spaceship.MaxSpeed
    # CHECK: lit.var.decl "speed"
    var speed: int = MySpeed

# // -----

# Test accessing an alias whose type is another alias, and materializing it.

trait Copyable:
    fn __copyinit__(out self, other: Self):
        ...

    fn copy(self) -> Self:
        return Self.__copyinit__(self)


trait ImplicitlyCopyable(Copyable):
    pass

struct ZInt(ImplicitlyCopyable):
    fn __init__(out self):
        pass
    fn __init__(out self, other: ZInt):
        pass

trait AnyType:
    pass

struct Spaceship:
    var fuel: ZInt

__extension Spaceship:
    alias InnerType: AnyType = ZInt
    alias MaxSpeed: InnerType = ZInt()

# CHECK-LABEL: lit.fn @"test_function()"
fn test_function():
    # This should work - accessing aliases from extensions
    # CHECK: lit.alias.decl *"MySpeed`"
    alias MySpeed = Spaceship.MaxSpeed
    # CHECK: lit.var.decl "speed"
    var speed = MySpeed

# // -----

# Test an extension method accessing and materializing an extension alias.

trait Copyable:
    fn __copyinit__(out self, other: Self):
        ...

    fn copy(self) -> Self:
        return Self.__copyinit__(self)


trait ImplicitlyCopyable(Copyable):
    pass

struct ZInt(ImplicitlyCopyable):
    fn __init__(out self):
        pass
    fn __init__(out self, other: ZInt):
        pass

struct Spaceship:
    var fuel: ZInt

__extension Spaceship:
    alias MaxSpeed = ZInt()

    # CHECK-LABEL: lit.fn @"get_max_speed
    fn get_max_speed(self: Spaceship) -> ZInt:
        # CHECK: %0 = kgen.param.materialize: !ZInt =
        return ZInt(MaxSpeed)
