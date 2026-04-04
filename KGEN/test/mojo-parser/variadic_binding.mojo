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
    def __getitem_param__[idx: Int](ref self) -> ref [self] Self.elt_types[idx]:
        pass


def only_copyable[T: Copyable](t: T):
    pass


# CHECK: @"f0{{.*}}"<elt_type: param_list<!Copyable>{{.*}}(%t: !lit.ref<!lit.struct<#SomeVA <:param_list<!AnyType> upcast(:param_list<!Copyable> elt_type)>
def f0[*elt_type: Copyable](t: SomeVA[*elt_type]):
    # Should be able to call only_copyable without an downcast.

    # CHECK lit.call @variadic_binding::@"only_copyable
    only_copyable(t[0])
    pass


# CHECK-LABEL: lit.fn @"f1
# CHECK-SAME: <elt_type: param_list<!mt_Int>{{.*}}(%t: !lit.ref<!lit.struct<#SomeVA <:param_list<!AnyType> upcast(:param_list<!mt_Int> elt_type)>
def f1[*elt_type: type_of(Int)](t: SomeVA[*elt_type]):
    pass

# CHECK-LABEL: lit.fn @"foo
def foo():
    # CHECK: lit.call {{.*}}::@"f0{{.*}}"[{{.*}}]<:param_list<!Copyable> [!SomeCopyable, !SomeCopyable]>
    f0(SomeVA[SomeCopyable, SomeCopyable]())

    # CHECK: lit.call {{.*}}::@"f1{{.*}}"[{{.*}}]<:param_list<!mt_Int> [!Int, !Int]>
    f1(SomeVA[Int, Int]())
