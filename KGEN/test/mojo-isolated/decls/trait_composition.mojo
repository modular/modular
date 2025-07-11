# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values -split-input-file | FileCheck %s

# LIT dialect asm aliases for trait composition.
# CHECK-DAG: !Trait1 = !lit.trait<@trait_composition::@Trait1>
# CHECK-DAG: !Trait2 = !lit.trait<@trait_composition::@Trait2>
# CHECK-DAG: !Trait3 = !lit.trait<@trait_composition::@Trait3>
# CHECK-DAG: !Trait1_Trait2 = !lit.trait<@trait_composition::@Trait1, @trait_composition::@Trait2>
# CHECK-DAG: !Trait1_Trait2_Trait3 = !lit.trait<@trait_composition::@Trait1, @trait_composition::@Trait2, @trait_composition::@Trait3>


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


@fieldwise_init
struct Struct123(Trait1, Trait2, Trait3):
    fn f1(self):
        pass

    fn f2(self):
        pass

    fn f3(self):
        pass


# Use direct trait union as parent.
@fieldwise_init
struct Struct12Direct(Trait1 & Trait2):
    fn f1(self):
        pass

    fn f2(self):
        pass


# Use trait union alias.
@fieldwise_init
struct Struct12Alias(Traits12):
    fn f1(self):
        pass

    fn f2(self):
        pass


fn useAny[T: AnyType](x: T):
    pass


# CHECK: lit.fn @"use1
# CHECK-SAME: <T: !Trait1>
# CHECK-SAME: (%x: !lit.ref<:!Trait1 T,
fn use1[T: Trait1](x: T):
    # CHECK: lit.call[{{.*}}"self": !lit.ref<:!Trait1 T,{{.*}} #kgen.get_witness<:!Trait1 T, "trait_composition::Trait1", "f1">][{{.*}}](%x)
    x.f1()


fn use2[T: Trait2](x: T):
    x.f2()


# Use aliased trait composition.
# CHECK: lit.fn @"use12
# CHECK-SAME: <T: !Trait1_Trait2>
# CHECK-SAME: (%x: !lit.ref<:!Trait1_Trait2 T,
fn use12[T: Traits12](x: T):
    # CHECK: lit.call @trait_composition::@"use1
    # CHECK-SAME: <:!Trait1 !kgen.param<:!Trait1_Trait2 T>>
    use1[T](x)
    # CHECK: lit.call @trait_composition::@"use2
    # CHECK-SAME: <:!Trait2 !kgen.param<:!Trait1_Trait2 T>>
    use2[T](x)


# Use direct trait composition.
fn use23[T: Trait2 & Trait3](x: T):
    # CHECK: lit.call[
    # CHECK-SAME: "self": !lit.ref<:!Trait2_Trait3 T,
    # CHECK-SAME: #kgen.get_witness<:!Trait3 !kgen.param<:!Trait2_Trait3 T>, "trait_composition::Trait3", "f3">
    x.f3()


fn use123[T: Traits123](x: T):
    # CHECK: lit.call @trait_composition::@"use23
    # CHECK-SAME: "x": !lit.ref<:!Trait1_Trait2_Trait3 T,
    use23(x)


# CHECK: lit.fn @"main_use()"
fn main_use():
    s123 = Struct123()

    # CHECK: lit.call @trait_composition::@"useAny
    # CHECK-SAME: <:!AnyType !Struct123>
    useAny(s123)
    # CHECK: lit.call @trait_composition::@"use1
    # CHECK-SAME: <:!Trait1 !Struct123>
    use1(s123)
    # CHECK: lit.call @trait_composition::@"use1
    # CHECK-SAME: <:!Trait1 !Struct123>
    use1[Struct123](s123)
    # CHECK: lit.call @trait_composition::@"use12
    # CHECK-SAME: <:!Trait1_Trait2 !Struct123>
    use12(s123)
    # CHECK: lit.call @trait_composition::@"use23
    # CHECK-SAME: <:!Trait2_Trait3 !Struct123>
    use23(s123)
    # CHECK: lit.call @trait_composition::@"use123
    # CHECK-SAME: <:!Trait1_Trait2_Trait3 !Struct123>
    use123(s123)

    s12direct = Struct12Direct()
    # CHECK: lit.call @trait_composition::@"use12
    # CHECK-SAME: <:!Trait1_Trait2 !Struct12Direct>
    use12(s12direct)
    # CHECK: lit.call @trait_composition::@"use12
    # CHECK-SAME: <:!Trait1_Trait2 !Struct12Direct>
    use12[Struct12Direct](s12direct)

    s12alias = Struct12Alias()
    # CHECK: lit.call @trait_composition::@"use12
    # CHECK-SAME: <:!Trait1_Trait2 !Struct12Alias>
    use12(s12alias)
    # CHECK: lit.call @trait_composition::@"use12
    # CHECK-SAME: <:!Trait1_Trait2 !Struct12Alias>
    use12[Struct12Alias](s12alias)


# // -----

# Test conditional method that refines the self type to a different trait.


trait Trait1:
    fn f1(self):
        ...


trait Trait2:
    fn f2(self):
        ...


alias Traits12 = Trait1 & Trait2


@fieldwise_init
struct Struct12(Traits12):
    fn f1(self):
        pass

    fn f2(self):
        pass


# conditional method
@fieldwise_init
struct Wrapper[T: AnyType]:
    fn cond1[T: Trait1](self: Wrapper[T], other: Wrapper[T]):
        pass


# CHECK: lit.fn @"useCond1
fn useCond1[
    ElementType: Traits12
](p1: Wrapper[ElementType], p2: Wrapper[ElementType]):
    # CHECK: lit.call {{.*}}@Wrapper::@"cond1
    # CHECK-SAME: <:!AnyType !kgen.param<:!Trait1_Trait2 ElementType>, :!Trait1 !kgen.param<:!Trait1_Trait2 ElementType>>
    p1.cond1(p2)


# // -----

# Check that constructor calls work with trait compositions.


trait Defaultable:
    fn __init__(out self):
        ...


trait IntConstructable:
    fn __init__(out self, x: Int):
        ...


# CHECK-LABEL: lit.fn @"useIntConstructable
fn useIntConstructable[T: Defaultable & IntConstructable]() -> T:
    # CHECK: %[[INT33:.*]] = {{.*}} !Int = <{33}>
    # CHECK: lit.call[
    # CHECK-SAME: #kgen.get_witness<:!IntConstructable !kgen.param<:!Defaultable_IntConstructable T>, "trait_composition::IntConstructable", "__init__">
    # CHECK-SAME: %[[INT33]]
    return T(33)


@register_passable("trivial")
struct MyStruct(Defaultable, IntConstructable):
    var x: Int

    fn __init__(out self):
        self.x = 42

    fn __init__(out self, x: Int):
        self.x = x


# // -----

# Check that we can call parametric trait methods on types that were declared
# with trait composition.


trait Writer:
    fn write(self):
        ...


trait Writable:
    fn write_to[T: Writer](self, x: T):
        ...


trait Defaultable:
    fn __init__(out self):
        ...


struct YourStruct:
    var x: Int

    fn __init__(out self):
        self.x = 42

    fn foo[W: Writable](self, x: W):
        pass

    fn do_it[W: Writable & Defaultable](self, x: W):
        self.foo(x)  # make sure this doesn't crash


# // -----

# Check that composition works with trait parameters.


trait Trait1:
    fn f1(self):
        ...


trait Trait2:
    fn f2(self):
        ...


trait Trait1C(Trait1):
    fn f1C(self):
        ...


@fieldwise_init
struct Struct1C(Trait1C, Trait2):
    fn f1(self):
        pass

    fn f2(self):
        pass

    fn f1C(self):
        pass


fn trait_param[A: __type_of(Trait1), T: A & Trait2](x: T):
    x.f1()
    x.f2()


fn use_trait_param():
    s1c = Struct1C()
    trait_param[Trait1C, Struct1C](s1c)
