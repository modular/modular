# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s

from builtin.device_passable import DevicePassable

trait MyInterface:
    fn thing(self):
        ...


fn make_closure(x: Int) -> Int:
    fn parametric[T: MyInterface](a: T) unified {}:
        # expected-error @below {{use of unknown declaration 'A'}}
        comptime X = A
        pass

    return x

struct Mem(ImplicitlyCopyable):
    pass

fn use(a:Mem):
    pass


fn foo(a: Mem):
    # expected-error @below {{cannot capture a by copy or move because it is not register passable and your closure is marked as register passable.}}
    fn closure() unified register_passable {var}:
        use(a)


fn bar(a: Mem):
    # expected-error @below {{a function cannot be register passable unless it is unified}}
    fn closure() register_passable {var}:
        use(a)

# COM: ambiguous captures

fn aThing(x: Int) -> Int:
    return x


fn aThing() -> Int:
    return 5


fn definesClosure():
    # expected-error @below {{ambiguous captured value: 'aThing'}}
    fn aClosure() unified {var aThing}:
        pass


struct Bar(ImplicitlyCopyable, RegisterType):
    var x: Int
    var y: Int

    fn __copyinit__(out self, other: Self):
        pass

fn takeDevicePassable[T: DevicePassable](impl: T):
    pass


def foo(bar: Bar):
    # COM: This should fail because Bar is not trivial.

    fn closure(number: Int) unified register_passable {var bar} -> Int:
        return bar.x

    # TODO: Rename Wrappers (MOCO-2541)
    # expected-error @below {{cannot bind type 'fn(number: Int) -> Int_Mova_Impl_Copy_Impl[__mlir_type.`!kgen.closure<@"unified_closures_errors::foo(unified_closures_errors::Bar)", "closure" register_passable>`, {}]' to trait 'DevicePassable'}}
    takeDevicePassable[type_of(closure)](closure)


# COM: Test that a register_passable closure capturing a non trivial
# COM: register_passable type does NOT conform to TrivialRegisterType.
# expected-note @below {{function declared here}}
fn takeTrivialRegisterType[T: TrivialRegisterType](impl: T):
    pass


def testNonTrivialClosureNotTrivialRegisterType(bar: Bar):
    fn closure() unified register_passable {var bar} -> Int:
        return bar.x

    # expected-error-re @below {{'{{.*}}' does not conform to trait 'TrivialRegisterType'}}
    takeTrivialRegisterType(closure)


# expected-note @below {{function declared here}}
fn changeIt(mut aString: String):
    pass


def nestedCaptureAll(mut aString: String):
    fn aFinalThing(x:Int) unified {read}:
        # expected-error @below {{invalid call to 'changeIt': value passed to mutable argument 'aString' must be mutable}}
        changeIt(aString)

        fn aChildThing(x:Int) unified {var}:
            changeIt(aString)



fn topLevel(x: String) -> String:
    return x


fn takesClosure[T: fn(Int) unified -> Int](cb: T, x: Int) -> Int:
    return cb(x)


fn useTopLevelClosure():
    # expected-error @below {{cannot convert 'fn(x: String) -> String' to trait 'fn(Int) -> Int'}}
    takesClosure[topLevel](topLevel, 1)


# ===----------------------------------------------------------------------=== #
# Closure type mismatch errors
# ===----------------------------------------------------------------------=== #

trait Animal:
    fn speak(self):
        ...


trait Mammal(Animal):
    pass

struct Dog(Mammal):
    fn speak(self):
        pass

# expected-note @below {{function declared here}}
fn takeClosureMammalParam[W: Mammal, C: fn (x: W) unified -> None](impl: C):
    pass


fn traitConstraintMismatch[Q: Animal]():
    fn closure(x: Q) unified {var}:
        x.speak()

    # expected-error @below {{does not conform to trait 'fn(x: W) -> None'}}
    takeClosureMammalParam(closure)

    fn closureWrongConvention(mut x: Dog) unified {var}:
        x.speak()

    # expected-error-re @below {{cannot bind type '{{.*}}' to trait 'fn(x: W) -> None'}}
    takeClosureMammalParam[Dog, type_of(closureWrongConvention)](closureWrongConvention)

# ===----------------------------------------------------------------------=== #
# Enforce Parameter Capture
# ===----------------------------------------------------------------------=== #

trait Coord(ImplicitlyCopyable):
  pass

struct Cartesian(Coord):
   var x: Int
   var y: Int
   var z: Int

struct Sphere(Coord):
   var theta: Int
   var phi: Int


# expected-note @below {{constraint declared here evaluated to False}}
# expected-note @below {{function declared here}}
fn takeClosure[T: Coord, C:fn() unified -> T](impl: C) -> T:
   _ = impl()


fn makeClosure[B:Int](something: Cartesian):
   fn closureImpl() unified {var} -> Cartesian:
      return something
   # expected-error @below {{invalid call to 'takeClosure': violated constraint}}
   takeClosure[Sphere, type_of(closureImpl)](closureImpl)
