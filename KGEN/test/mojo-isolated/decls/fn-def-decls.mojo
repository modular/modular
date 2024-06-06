# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK-LABEL: lit.func @"empty_def()"() -> !kgen.none
# CHECK: lit.end_func
fn empty_def():
    pass


# CHECK-LABEL: lit.func @"slash
# CHECK-SAME: (%a: index borrow, |, %b: index borrow)
fn slash(a: int, /, b: int):
    pass


# CHECK-LABEL: lit.func @"trailing_slash
# CHECK-SAME: (%a: index borrow, |)
fn trailing_slash(a: int, /):
    pass


# CHECK-LABEL: lit.func @"star
# CHECK-SAME: (%a: index borrow, *, %b: index borrow)
fn star(a: int, *, b: int):
    pass


# CHECK-LABEL: lit.func @"leading_star
# CHECK-SAME: (*, %a: index borrow)
fn leading_star(*, a: int):
    pass


# CHECK-LABEL: lit.func @"star_and_slash
# CHECK-LABEL: (%a: index borrow, |, *, %b: index borrow)
fn star_and_slash(a: int, /, *, b: int):
    pass


# CHECK-LABEL: lit.func @"star_and_slash_2
# CHECK-SAME: (%a: index borrow, |, %b: index borrow, *, %c: index borrow)
fn star_and_slash_2(a: int, /, b: int, *, c: int):
    pass


# CHECK-LABEL: lit.func @"default_args
# CHECK-SAME: (%a: index borrow, %b: index borrow = 8, *, %c: index borrow, %d: index borrow = 9)
fn default_args(a: int, b: int = `8`, *, c: int, d: int = `9`):
    pass


# CHECK-LABEL: lit.func @"variadic_and_kw_only
# CHECK-SAME: (%a: index borrow, %b: index borrow, %args: !kgen.variadic<index> borrow|var, *, %c: index borrow, %d: index borrow = 9)
fn variadic_and_kw_only(a: int, b: int, *args: int, c: int, d: int = `9`):
    pass


# CHECK-LABEL: lit.func @"variadic_arg_after_default
# CHECK-SAME: (%a: index borrow, %b: index borrow = 0, %args: !kgen.variadic<index> borrow|var = *?,
# CHECK-SAME:  *, %c: index borrow, %d: index borrow = 1, %kwargs: {{.*}}|var = *?)
fn variadic_arg_after_default(
    a: int, b: int = `0`, *args: int, c: int, d: int = `1`, **kwargs: int
):
    pass


# CHECK-LABEL: lit.func @"variadic_param_after_default
# CHECK-SAME: <a, b = 0, args: {{.*}} var = *?, *, c, d = 1>()
fn variadic_param_after_default[
    a: int, b: int = `0`, *args: int, c: int, d: int = `1`
]():
    pass


# CHECK-LABEL: lit.func @"inferred_params
# CHECK-SAME: <x, y, +>
fn inferred_params[x: int, y: int, //]():
    # CHECK-NEXT: !lit.signature<<"x": index, "y": index, +>() -> !kgen.none> = <@"
    alias fn_type: fn[x: int, y: int, //]() -> None = inferred_params


# CHECK-LABEL: lit.func @"inferred_params_regular
# CHECK-SAME: <x, +, y>
fn inferred_params_regular[x: int, //, y: int]():
    # CHECK-NEXT: !lit.signature<<"x": index, +, "y": index>() -> !kgen.none> = <@"
    alias fn_type: fn[x: int, //, y: int]() -> None = inferred_params_regular


# CHECK-LABEL: lit.func @"inferred_params_pos_only
# CHECK-SAME: <x, +, y = 1, |>
fn inferred_params_pos_only[x: int, //, y: int = `1`, /]():
    pass


# CHECK-LABEL: lit.func @"inferred_params_kw_only
# CHECK-SAME: <x, +, *, y>
fn inferred_params_kw_only[x: int, //, *, y: int]():
    pass

# ===----------------------------------------------------------------------=== #
# Test that def arguments are assignable and we get the right number of copies.
# ===----------------------------------------------------------------------=== #

@value
struct MemoryOnly:
    pass
@value
@register_passable
struct NonTrivialReg:
    pass



# CHECK-LABEL: lit.func @"defTests({{.*}}, %untyped: !lit.ref<!object, imm {{.*}}> borrow_in_mem,
def defTests(a: Int, b: Int, mem: MemoryOnly, reg: NonTrivialReg, untyped) -> None:
  # CHECK-NEXT: lit.var.decl "defTests"

  # CHECK-NEXT: %reg_0 = lit.var.decl "reg" arg(3)
  # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}NonTrivialReg::@"__copyinit__{{.*}}(%reg)
  # CHECK-NEXT: lit.ref.store [[TMP]], %reg_0

  # CHECK-NEXT: %mem_1 = lit.var.decl "mem" arg(2)
  # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__copyinit__{{.*}}(%mem_1, %mem)

  # CHECK-NEXT: %a_2 = lit.var.decl "a" arg
  # CHECK-NEXT: lit.ref.store %a, %a_2

  # CHECK-NEXT: lit.ref.store %b, %a_2
  a = b # Arguments are mutable!

  # CHECK-NEXT: lit.ref.store %b, %a_2
  a = b # Subsequent arguments don't re-make the box.

  # CHECK-NEXT: [[TMP:%.*]] = kgen.param.materialize: !MemoryOnly = <{}>
  # CHECK-NEXT: lit.ref.store [[TMP]], %mem_1
  mem = MemoryOnly()

  # CHECK-NEXT: [[TMP:%.*]] = kgen.param.materialize: !NonTrivialReg = <{}>
  # CHECK-NEXT: lit.ref.store [[TMP]], %reg_0
  reg = NonTrivialReg()

  # Issue#38762
  # MOCO-83: [mojo][Bug] def methods can't shadow names via assignment
  defTests = 4

struct TypeWithParametricSelf:
    fn method(ref[_] self: Self): pass

struct ValueWithTypeWithParametricSelf:
    var member: TypeWithParametricSelf

# CHECK-LABEL: test_def_arg_box_mbvalue
def test_def_arg_box_mbvalue(a: TypeWithParametricSelf, b: ValueWithTypeWithParametricSelf):
    # CHECK-NEXT: %xyz = lit.var.decl "xyz"
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}method{{.*}}(%a)
    # CHECK-NEXT: lit.ref.store [[TMP]], %xyz
    var xyz = a.method()

    # MOCO-715: failed to infer implicit parameter 'is_mutable' of argument 'self' type 'Reference
    # CHECK-NEXT: [[MEMBERREF:%.*]] = lit.ref.struct.ger %b[member]
    _ = b.member.method()

fn returnsMultiple() -> (Int, MemoryOnly): pass


# MOCO-687: Unable to destructure multiple outputs in a def function without explicit var declarations
# CHECK-LABEL: test_multi_tuple_def_value
def test_multi_tuple_def_value():
     # CHECK: %b = lit.var.decl "b"
     # CHECK: %a = lit.var.decl "a"
     a, b = returnsMultiple()

# CHECK-LABEL: lit.func @"ref_result
fn ref_result(inout x: MemoryOnly) -> ref [__lifetime_of(x)] MemoryOnly:
    # CHECK-NEXT: lit.return %x : !lit.ref<!MemoryOnly, mut *"x`">
    return x

# CHECK-LABEL: lit.func @"def_ref_result
def def_ref_result(inout x: MemoryOnly) -> ref [__lifetime_of(x)] MemoryOnly:
    # CHECK-NEXT: lit.ref.store %x, %__result__
    # CHECK-NEXT: %0 = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return %0
    return x


# CHECK-LABEL: lit.func @"use_ref_result
def use_ref_result():
    # CHECK-NEXT: %a = lit.var.decl "a"
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.materialize: !MemoryOnly = <{}>
    # CHECK-NEXT: lit.ref.store [[TMP]], %a
    var a = MemoryOnly()

    # CHECK-NEXT: [[REF:%.*]] = lit.call {{.*}}ref_result{{.*}}(%a)
    ref_result(a) = MemoryOnly()
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.materialize: !MemoryOnly = <{}>
    # CHECK-NEXT: lit.ref.store [[TMP]], [[REF]]

    # CHECK-NEXT: %__call_result_tmp__ = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}decls"::@"def_ref_result{{.*}}(%a, %__error__, %__call_result_tmp__)
    # CHECK-NEXT: [[REF:%.*]] = lit.load.consume %__call_result_tmp__
    # CHECK-NEXT: [[TMP:%.*]] = kgen.param.materialize: !MemoryOnly = <{}>
    # CHECK-NEXT: lit.ref.store [[TMP]], [[REF]]
    def_ref_result(a) = MemoryOnly()

# CHECK-LABEL: lit.func @"return_def_arg_box
def return_def_arg_box(abc: MemoryOnly) -> ref [__lifetime_of(abc)] MemoryOnly:
# CHECK-NEXT: lit.ref.store %abc, %__result__
    return abc

