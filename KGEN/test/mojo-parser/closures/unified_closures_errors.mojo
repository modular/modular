# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %parse-mojo-isolated -verify-diagnostics %s

from std.builtin.device_passable import DevicePassable

trait MyInterface:
    def thing(self):
        ...


def make_closure(x: Int) -> Int:
    def parametric[T: MyInterface](a: T) unified {}:
        # expected-error @below {{use of unknown declaration 'A'}}
        comptime X = A
        pass

    return x

struct Mem(ImplicitlyCopyable):
    pass

def use(a:Mem):
    pass


def foo(a: Mem):
    # expected-error @below {{cannot capture a by copy or move because it is not register passable and your closure is marked as register passable}}
    def closure() unified register_passable {var}:
        use(a)


def bar(a: Mem):
    # expected-error @below {{'register_passable' functions must be 'unified'}}
    def closure() register_passable {var}:
        use(a)

# COM: ambiguous captures

def aThing(x: Int) -> Int:
    return x


def aThing() -> Int:
    return 5


def definesClosure():
    # expected-error @below {{ambiguous captured value: 'aThing'}}
    def aClosure() unified {var aThing}:
        pass


struct Bar(ImplicitlyCopyable, RegisterPassable):
    var x: Int
    var y: Int

    def __init__(out self, *, copy: Self):
        pass

# expected-note @+1 {{function declared here}}
def takeDevicePassable[T: DevicePassable](impl: T):
    pass


def foo(bar: Bar) raises:
    # COM: This should fail because Bar is not trivial.

    def closure(number: Int) unified register_passable {var bar} -> Int:
        return bar.x

    # TODO: Rename Wrappers (MOCO-2541)
    # expected-error @below {{'takeDevicePassable' parameter 'T' has 'DevicePassable' type, but value has type 'def(number: Int) register_passable -> Int'}}
    takeDevicePassable[type_of(closure)](closure)


# COM: Test that a register_passable closure capturing a non trivial
# COM: register_passable type does NOT conform to TrivialRegisterPassable.
# expected-note @below {{function declared here}}
def takeTrivialRegisterPassable[T: TrivialRegisterPassable](impl: T):
    pass


def testNonTrivialClosureNotTrivialRegisterPassable(bar: Bar) raises:
    def closure() unified register_passable {var bar} -> Int:
        return bar.x

    # expected-error-re @below {{'{{.*}}' does not conform to trait 'TrivialRegisterPassable'}}
    takeTrivialRegisterPassable(closure)


# expected-note @below {{function declared here}}
def changeIt(mut aString: String):
    pass


def nestedCaptureAll(mut aString: String) raises:
    def aFinalThing(x:Int) unified {read}:
        # expected-error @below {{invalid call to 'changeIt': value passed to mutable argument 'aString' must be mutable}}
        changeIt(aString)

        def aChildThing(x:Int) unified {var}:
            changeIt(aString)



def topLevel(x: String) -> String:
    return x

# expected-note @+1 {{function declared here}}
def takesClosure[T: def(Int) unified -> Int](cb: T, x: Int) -> Int:
    return cb(x)


def useTopLevelClosure():
    # expected-error @below {{invalid call to 'takesClosure': 'takesClosure' parameter 'T' has 'def(Int) -> Int' type, but value has type 'def topLevel(x: String) -> String'}}
    takesClosure[topLevel](topLevel, 1)


# ===----------------------------------------------------------------------=== #
# Closure type mismatch errors
# ===----------------------------------------------------------------------=== #

trait Animal:
    def speak(self):
        ...


trait Mammal(Animal):
    pass

struct Dog(Mammal):
    def speak(self):
        pass

# expected-note @below {{function declared here}}
def takeClosureMammalParam[W: Mammal, C: def (x: W) unified -> None](impl: C):
    pass


def traitConstraintMismatch[Q: Animal]():
    def closure(x: Q) unified {var}:
        x.speak()

    # expected-error @below {{does not conform to trait 'def(x: W) -> None'}}
    takeClosureMammalParam(closure)

    def closureWrongConvention(mut x: Dog) unified {var}:
        x.speak()

    # expected-error @below {{'takeClosureMammalParam' parameter 'C' has 'def(x: W) -> None' type, but value has type 'def(mut x: Dog) -> None'}}
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
def takeClosure[T: Coord, C:def() unified -> T](impl: C) -> T:
   _ = impl()


def makeClosure[B:Int](something: Cartesian):
   def closureImpl() unified {var} -> Cartesian:
      return something
   # expected-error @below {{invalid call to 'takeClosure': violated constraint}}
   takeClosure[Sphere, type_of(closureImpl)](closureImpl)


# ===----------------------------------------------------------------------=== #
# Non-compatible parameter signatures disqualify conformance
# ===----------------------------------------------------------------------=== #

def _print(x: Int):
    pass

# expected-note @below {{function declared here}}
def callee_no_params[
    func: def() unified -> None,
    //,
](closure: func):
    closure()


def incompatible_param_signature() raises:
    var x = 42

    @always_inline
    def my_func[param_only: Int]() unified {read x}:
        _print(x)

    # expected-error @below {{does not conform to trait 'def() -> None'}}
    callee_no_params(my_func)

# ===----------------------------------------------------------------------=== #
# Multiple default capture conventions specified
# ===----------------------------------------------------------------------=== #


def multiple_default_capture_conventions(x: Int):
    # expected-error @below {{default capture convention was already specified; remove the duplicate}}
    # expected-note @below {{a capture convention (like 'mut' or 'var') before the capture list sets the default for all captured variables}}
    def my_closure(y: Int) unified {var, ref} -> Int:
        return y

# ===----------------------------------------------------------------------=== #
# Incompatible capture conventions
# ===----------------------------------------------------------------------=== #


def incompatible_capture_conventions(x: Int):
    # expected-error @below {{'^' requires 'var' convention; write 'var x^' to move a capture}}
    def my_closure(y: Int) unified {ref x^} -> Int:
        return y


# ===----------------------------------------------------------------------=== #
# Default capture convention violation
# ===----------------------------------------------------------------------=== #

def default_capture_convention_violation():
    var y = 20
    var x = 10

    def my_fn() unified {read, mut y}:
        # Assigning to `y` work
        y = 20
        # expected-error @below {{expression must be mutable in assignment}}
        x = 10

# ===----------------------------------------------------------------------=== #
# Capture RTP with no-read convention
# ===----------------------------------------------------------------------=== #

def capture_RTP(x : Int) :
    # expected-error @below{{register passible value 'x' can not be captured by 'mut'. Do you mean 'read'?}}
    def my_func() unified {mut x}:
        pass
