# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


trait Trait1:
    fn f1(self):
        ...


trait Trait2:
    fn f2(self):
        ...


trait Trait3:
    fn f3(self):
        ...


alias Traits12 = Trait1 & Trait2
alias Traits123 = Trait1 & Trait2 & Trait3


@value
struct Struct123(Traits12, Trait3):
    fn f1(self):
        print("f1")

    fn f2(self):
        print("f2")

    fn f3(self):
        print("f3")


fn use1[T: Trait1](x: T):
    x.f1()


fn use2[T: Trait2](x: T):
    x.f2()


fn use12[T: Traits12](x: T):
    use1(x)
    use2(x)


fn use23[T: Trait2 & Trait3](x: T):
    x.f2()
    x.f3()


fn use123[T: Traits123](x: T):
    x.f1()
    use23(x)


# conditional method
@value
struct Wrapper[T: AnyType]:
    fn cond1[T: Trait1](self: Wrapper[T], other: Wrapper[T]):
        print("cond")


fn useCond1[
    ElementType: Traits12
](p1: Wrapper[ElementType], p2: Wrapper[ElementType]):
    p1.cond1(p2)


# constructor overloading
trait IntConstructable:
    fn __init__(out self, x: Int):
        ...


fn useIntConstructable[T: Defaultable & IntConstructable]() -> T:
    return T(33)


@register_passable("trivial")
struct MyStruct(Defaultable, IntConstructable):
    var x: Int

    fn __init__(out self):
        self.x = 42

    fn __init__(out self, x: Int):
        self.x = x


fn main():
    s123 = Struct123()

    # CHECK: f1
    use1(s123)

    # CHECK: f1
    # CHECK: f2
    use12(s123)

    # CHECK: f2
    # CHECK: f3
    use23(s123)

    # CHECK: f1
    # CHECK: f2
    # CHECK: f3
    use123(s123)

    # CHECK: cond
    useCond1(Wrapper[Struct123](), Wrapper[Struct123]())

    # CHECK: 33
    print(useIntConstructable[MyStruct]().x)
