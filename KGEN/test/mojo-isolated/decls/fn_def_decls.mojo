# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.fn @"empty_def()"() -> !kgen.none
# CHECK: lit.end_fn
fn empty_def():
    pass


# CHECK-LABEL: lit.fn @"slash
# CHECK-SAME: (%a: !Int, |, %b: !Int)
fn slash(a: Int, /, b: Int):
    pass


# CHECK-LABEL: lit.fn @"trailing_slash
# CHECK-SAME: (%a: !Int, |)
fn trailing_slash(a: Int, /):
    pass


# CHECK-LABEL: lit.fn @"star
# CHECK-SAME: (%a: !Int, *, %b: !Int)
fn star(a: Int, *, b: Int):
    pass


# CHECK-LABEL: lit.fn @"leading_star
# CHECK-SAME: (*, %a: !Int)
fn leading_star(*, a: Int):
    pass


# CHECK-LABEL: lit.fn @"star_and_slash
# CHECK-LABEL: (%a: !Int, |, *, %b: !Int)
fn star_and_slash(a: Int, /, *, b: Int):
    pass


# CHECK-LABEL: lit.fn @"star_and_slash_2
# CHECK-SAME: (%a: !Int, |, %b: !Int, *, %c: !Int)
fn star_and_slash_2(a: Int, /, b: Int, *, c: Int):
    pass


# CHECK-LABEL: lit.fn @"default_args
# CHECK-SAME: (%a: !Int, %b: !Int = {8}, *, %c: !Int, %d: !Int = {9})
fn default_args(a: Int, b: Int = 8, *, c: Int, d: Int = 9):
    pass


# CHECK-LABEL: lit.fn @"variadic_and_kw_only
# CHECK-SAME: (%a: !Int, %b: !Int, %args: !kgen.variadic<!Int> pos_vararg, *, %c: !Int, %d: !Int = {9})
fn variadic_and_kw_only(
    a: Int, b: Int, *args: Int, c: Int, d: Int = 9
):
    pass


# CHECK-LABEL: lit.fn @"variadic_arg_after_default
# CHECK-SAME: (%a: !Int, %b: !Int = {0}, %args: !kgen.variadic<!Int> pos_vararg = *?,
# CHECK-SAME:  *, %c: !Int, %d: !Int = {1}, %kwargs: {{.*}}|kw_vararg = *?)
fn variadic_arg_after_default(
    a: Int,
    b: Int = 0,
    *args: Int,
    c: Int,
    d: Int = 1,
    **kwargs: Int,
):
    pass


# CHECK-LABEL: lit.fn @"variadic_param_after_default
# CHECK-SAME: <a: !Int, b: !Int = {0}, args: {{.*}} pos_vararg = *?, *, c: !Int, d: !Int = {1}>()
fn variadic_param_after_default[
    a: Int, b: Int = 0, *args: Int, c: Int, d: Int = 1
]():
    pass


# CHECK-LABEL: lit.fn @"inferred_params
# CHECK-SAME: <x: !Int, y: !Int, +>
fn inferred_params[x: Int, y: Int, //]():
    # CHECK-NEXT: !lit.generator<<"x": !Int, "y": !Int, +>() -> !kgen.none> = <@
    alias fn_type: fn[x: Int, y: Int, //] () -> None = inferred_params


# CHECK-LABEL: lit.fn @"inferred_params_regular
# CHECK-SAME: <x: !Int, +, y: !Int>
fn inferred_params_regular[x: Int, //, y: Int]():
    # CHECK-NEXT: !lit.generator<<"x": !Int, +, "y": !Int>() -> !kgen.none> = <@
    alias fn_type: fn[
        x: Int, //, y: Int
    ] () -> None = inferred_params_regular


# CHECK-LABEL: lit.fn @"inferred_params_pos_only
# CHECK-SAME: <x: !Int, +, y: !Int = {1}, |>
fn inferred_params_pos_only[x: Int, //, y: Int = 1, /]():
    pass


# CHECK-LABEL: lit.fn @"inferred_params_kw_only
# CHECK-SAME: <x: !Int, +, *, y: !Int>
fn inferred_params_kw_only[x: Int, //, *, y: Int]():
    pass


# ===----------------------------------------------------------------------=== #
# Test that def arguments are assignable and we get the right number of copies.
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct MemoryOnly(ImplicitlyCopyable, Movable):
    pass


@fieldwise_init
@register_passable
struct NonTrivialReg(ImplicitlyCopyable):
    pass


struct TypeWithParametricSelf:
    fn method(ref self):
        pass


struct ValueWithTypeWithParametricSelf:
    var member: TypeWithParametricSelf


# CHECK-LABEL: test_def_arg_box_mbvalue
def test_def_arg_box_mbvalue(
    a: TypeWithParametricSelf, b: ValueWithTypeWithParametricSelf
):
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}method{{.*}}(%a)
    # CHECK-NEXT: %xyz = lit.var.decl "xyz"
    # CHECK-NEXT: lit.ref.store [[TMP]], %xyz
    var xyz = a.method()

    # MOCO-715: failed to infer implicit parameter 'mut' of argument 'self' type 'Pointer
    # CHECK-NEXT: [[MEMBERREF:%.*]] = lit.ref.struct.ger %b[member]
    _ = b.member.method()


fn returnsMultiple() -> Tuple[Int, MemoryOnly]:
    pass


# MOCO-687: Unable to destructure multiple outputs in a def function without explicit var declarations
# CHECK-LABEL: test_multi_tuple_def_value
def test_multi_tuple_def_value():
    # CHECK: %b = lit.var.decl "b"
    # CHECK: %a = lit.var.decl "a"
    a, b = returnsMultiple()


# CHECK-LABEL: lit.fn @"ref_result
fn ref_result(mut x: MemoryOnly) -> ref [x] MemoryOnly:
    # CHECK-NEXT: lit.return %x : !lit.ref<!MemoryOnly, mut *"x`">
    return x


# CHECK-LABEL: lit.fn @"def_ref_result
def def_ref_result(mut x: MemoryOnly) -> ref [x] MemoryOnly:
    # CHECK-NEXT: lit.ref.store %x, %__result__
    # CHECK-NEXT: %0 = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return %0
    return x


# CHECK-LABEL: lit.fn @"use_ref_result
def use_ref_result():
    # CHECK-NEXT: %a = lit.var.decl "a"
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}(%a)
    var a = MemoryOnly()

    # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}ref_result{{.*}}(%a)
    ref_result(a) = MemoryOnly()
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}([[REF]])

    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}decls::@"def_ref_result{{.*}}(%a, %__error__, %__call_result_tmp__)
    # CHECK-NEXT: [[REF:%.*]] = lit.load.consume %__call_result_tmp__
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}([[REF]])
    def_ref_result(a) = MemoryOnly()


# CHECK-LABEL: lit.fn @"return_def_arg_box
def return_def_arg_box(abc: MemoryOnly) -> ref [abc] MemoryOnly:
    # CHECK-NEXT: lit.ref.store %abc, %__result__
    return abc


# CHECK-LABEL: lit.fn @"foldable_requires_1
# CHECK-SAME: where {
# CHECK-SAME:   ne(#lit.struct.extract<:!Int x, "_mlir_value">, 0){{.*}}>}
fn foldable_requires_1[x: Int]()
    where x:
        pass


# CHECK-LABEL: lit.fn @"foldable_requires_2
# CHECK-SAME: where {
# CHECK-SAME:   ge(#lit.struct.extract<:!Int y, "_mlir_value">, 11)
# CHECK-SAME:   lt(#lit.struct.extract<:!Int x, "_mlir_value">, {{.*}}
fn foldable_requires_2[x: Int, y: Int]()
    where y > 10
    where x < 1:
        pass


# CHECK-LABEL: lit.fn @"foldable_requires_passthru
fn foldable_requires_passthru[a: Int, b: Int]()
    where a > 10  # test comment
    where b < 1
    where b       # test another comment
    where a:      # test another comment
        foldable_requires_2[b, a]()
        foldable_requires_1[a]()
        foldable_requires_1[b]()


# CHECK-LABEL: lit.fn @"foldable_requires_param_if
fn foldable_requires_param_if[a: Int, b: Int]():
    @parameter
    if a > 10:
        @parameter
        if b < 1:
            foldable_requires_2[b, a]()
    elif a < 1:
        @parameter
        if b > 10:
            foldable_requires_2[a, b]()
    elif a:
        foldable_requires_1[a]()
    elif b:
        foldable_requires_1[b]()
    else:
        foldable_requires_1[42]()


# CHECK-LABEL: lit.fn @"foldable_param_requires_1
# CHECK-SAME: <x: !Int {{.*}}ne(#lit.struct.extract<:!Int x, "_mlir_value">, 0)
fn foldable_param_requires_1[x: Int where x]():
    pass


# CHECK-LABEL: lit.fn @"foldable_param_requires_2
# CHECK-SAME: <x: !Int {{.*}}lt(#lit.struct.extract<:!Int x, "_mlir_value">, 1)
# CHECK-SAME:  y: !Int {{.*}}ge(#lit.struct.extract<:!Int y, "_mlir_value">, 11)
fn foldable_param_requires_2[
    x: Int where x < 1,
    y: Int where y > 10 = 11
]():
    pass
