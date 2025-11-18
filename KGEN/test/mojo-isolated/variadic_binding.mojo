# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


@fieldwise_init
struct SomeCopyable(Copyable):
    pass


@fieldwise_init
struct SomeVA[*elt_types: AnyType]:
    fn __getitem__[idx: Int](ref self) -> ref [self] Self.elt_types[idx]:
        pass


fn only_copyable[T: Copyable](t: T):
    pass


# CHECK: @"f0{{.*}}"<elt_type: variadic<!Copyable>{{.*}}(%t: !lit.ref<@{{.*}}::@SomeVA<:variadic<!AnyType> upcast(:variadic<!Copyable> elt_type)>
fn f0[*elt_type: Copyable](t: SomeVA[*elt_type]):
    # Should be able to call only_copyable without an downcast.

    # CHECK lit.call @variadic_binding::@"only_copyable
    only_copyable(t[0])
    pass


# CHECK: @"f1{{.*}}"<elt_type: variadic<!mt_Int>{{.*}}(%t: !lit.ref<@{{.*}}::@SomeVA<:variadic<!AnyType> upcast(:variadic<!mt_Int> elt_type)>
fn f1[*elt_type: type_of(Int)](t: SomeVA[*elt_type]):
    pass


fn foo():
    # CHECK: lit.call @{{.*}}::@"f0{{.*}}"[{{.*}}]<:variadic<!Copyable> [!SomeCopyable, !SomeCopyable]>
    f0(SomeVA[SomeCopyable, SomeCopyable]())

    # CHECK: lit.call @{{.*}}::@"f1{{.*}}"[{{.*}}]<:variadic<!mt_Int> [!Int, !Int]>
    f1(SomeVA[Int, Int]())
