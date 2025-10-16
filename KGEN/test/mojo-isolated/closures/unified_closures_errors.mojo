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
        alias X = A
        pass

    return x

struct Mem(Copyable, ImplicitlyCopyable):
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
    # expected-error @below {{ambiguous captured value: aThing}}
    fn aClosure() unified {var aThing}:
        pass




@register_passable
struct Bar(ImplicitlyCopyable):
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
    # expected-error @below {{cannot bind type 'fn(number: Int) -> Int_wrapper_copyable[!kgen.closure<_unified_closures_errors::_"foo(unified_closures_errors::Bar)", "closure" register_passable>, {}]' to trait 'DevicePassable'}}
    takeDevicePassable[type_of(closure)](closure)
