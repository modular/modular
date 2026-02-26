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


struct Bar(ImplicitlyCopyable, RegisterPassable):
    var x: Int
    var y: Int

    fn __init__(out self, *, copy: Self):
        pass

# expected-note @+1 {{function declared here}}
fn takeDevicePassable[T: DevicePassable](impl: T):
    pass


def foo(bar: Bar):
    # COM: This should fail because Bar is not trivial.

    fn closure(number: Int) unified register_passable {var bar} -> Int:
        return bar.x

    # TODO: Rename Wrappers (MOCO-2541)
    # expected-error @below {{'takeDevicePassable' parameter 'T' has 'DevicePassable' type, but value has type 'AnyStruct[fn(number: Int) -> Int_Mova_Impl_Copy_Impl[__mlir_type.`!kgen.closure<@"unified_closures_errors::foo(unified_closures_errors::Bar)", "closure" register_passable>`, {}]]'}}
    takeDevicePassable[type_of(closure)](closure)


# COM: Test that a register_passable closure capturing a non trivial
# COM: register_passable type does NOT conform to TrivialRegisterPassable.
# expected-note @below {{function declared here}}
fn takeTrivialRegisterPassable[T: TrivialRegisterPassable](impl: T):
    pass


def testNonTrivialClosureNotTrivialRegisterPassable(bar: Bar):
    fn closure() unified register_passable {var bar} -> Int:
        return bar.x

    # expected-error-re @below {{'{{.*}}' does not conform to trait 'TrivialRegisterPassable'}}
    takeTrivialRegisterPassable(closure)


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

# expected-note @+1 {{function declared here}}
fn takesClosure[T: fn(Int) unified -> Int](cb: T, x: Int) -> Int:
    return cb(x)


fn useTopLevelClosure():
    # expected-error @below {{invalid call to 'takesClosure': 'takesClosure' parameter 'T' has 'fn(Int) -> Int' type, but value has type 'fn(x: String) -> String'}}
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

    # expected-error-re @below {{'takeClosureMammalParam' parameter 'C' has 'fn(x: W) -> None' type, but value has type 'AnyStruct[fn(mut x: Dog) -> None_Mova_Impl_Copy_Impl[__mlir_type.`!kgen.closure<@"unified_closures_errors::traitConstraintMismatch[unified_closures_errors::Animal]()", "closureWrongConvention" nonescaping>`, {}]]'}}
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


# ===----------------------------------------------------------------------=== #
# Non-compatible parameter signatures disqualify conformance
# ===----------------------------------------------------------------------=== #

fn _print(x: Int):
    pass

# expected-note @below {{function declared here}}
fn callee_no_params[
    func: fn() unified -> None,
    //,
](closure: func):
    closure()


def incompatible_param_signature():
    var x = 42

    @always_inline
    fn my_func[param_only: Int]() unified {read x}:
        _print(x)

    # expected-error @below {{does not conform to trait 'fn() -> None'}}
    callee_no_params(my_func)
