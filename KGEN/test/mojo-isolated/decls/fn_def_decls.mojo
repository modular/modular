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
# CHECK-SAME: (%a: index, |, %b: index)
fn slash(a: Index, /, b: Index):
    pass


# CHECK-LABEL: lit.fn @"trailing_slash
# CHECK-SAME: (%a: index, |)
fn trailing_slash(a: Index, /):
    pass


# CHECK-LABEL: lit.fn @"star
# CHECK-SAME: (%a: index, *, %b: index)
fn star(a: Index, *, b: Index):
    pass


# CHECK-LABEL: lit.fn @"leading_star
# CHECK-SAME: (*, %a: index)
fn leading_star(*, a: Index):
    pass


# CHECK-LABEL: lit.fn @"star_and_slash
# CHECK-LABEL: (%a: index, |, *, %b: index)
fn star_and_slash(a: Index, /, *, b: Index):
    pass


# CHECK-LABEL: lit.fn @"star_and_slash_2
# CHECK-SAME: (%a: index, |, %b: index, *, %c: index)
fn star_and_slash_2(a: Index, /, b: Index, *, c: Index):
    pass


# CHECK-LABEL: lit.fn @"default_args
# CHECK-SAME: (%a: index, %b: index = 8, *, %c: index, %d: index = 9)
fn default_args(a: Index, b: Index = `8`, *, c: Index, d: Index = `9`):
    pass


# CHECK-LABEL: lit.fn @"variadic_and_kw_only
# CHECK-SAME: (%a: index, %b: index, %args: !kgen.variadic<index> var, *, %c: index, %d: index = 9)
fn variadic_and_kw_only(a: Index, b: Index, *args: Index, c: Index, d: Index = `9`):
    pass


# CHECK-LABEL: lit.fn @"variadic_arg_after_default
# CHECK-SAME: (%a: index, %b: index = 0, %args: !kgen.variadic<index> var = *?,
# CHECK-SAME:  *, %c: index, %d: index = 1, %kwargs: {{.*}}|var = *?)
fn variadic_arg_after_default(
    a: Index, b: Index = `0`, *args: Index, c: Index, d: Index = `1`, **kwargs: Index
):
    pass


# CHECK-LABEL: lit.fn @"variadic_param_after_default
# CHECK-SAME: <a, b = 0, args: {{.*}} var = *?, *, c, d = 1>()
fn variadic_param_after_default[
    a: Index, b: Index = `0`, *args: Index, c: Index, d: Index = `1`
]():
    pass


# CHECK-LABEL: lit.fn @"inferred_params
# CHECK-SAME: <x, y, +>
fn inferred_params[x: Index, y: Index, //]():
    # CHECK-NEXT: !lit.generator<<"x": index, "y": index, +>() -> !kgen.none> = <@
    alias fn_type: fn[x: Index, y: Index, //] () -> None = inferred_params


# CHECK-LABEL: lit.fn @"inferred_params_regular
# CHECK-SAME: <x, +, y>
fn inferred_params_regular[x: Index, //, y: Index]():
    # CHECK-NEXT: !lit.generator<<"x": index, +, "y": index>() -> !kgen.none> = <@
    alias fn_type: fn[x: Index, //, y: Index] () -> None = inferred_params_regular


# CHECK-LABEL: lit.fn @"inferred_params_pos_only
# CHECK-SAME: <x, +, y = 1, |>
fn inferred_params_pos_only[x: Index, //, y: Index = `1`, /]():
    pass


# CHECK-LABEL: lit.fn @"inferred_params_kw_only
# CHECK-SAME: <x, +, *, y>
fn inferred_params_kw_only[x: Index, //, *, y: Index]():
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


# CHECK-LABEL: lit.fn @"defTests({{.*}},
def defTests(
    a: Int, b: Int, mem: MemoryOnly, reg: NonTrivialReg
) -> None:
    # CHECK-NEXT: lit.var.decl "defTests"

    # CHECK-NEXT: %reg_0 = lit.var.decl "reg" arg(3)
    # CHECK-NEXT: lit.call {{.*}}NonTrivialReg::@"__copyinit__{{.*}}(%reg, %reg_0)

    # CHECK-NEXT: %mem_1 = lit.var.decl "mem" arg(2)
    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__copyinit__{{.*}}(%mem, %mem_1)

    # CHECK-NEXT: %a_2 = lit.var.decl "a" arg
    # CHECK-NEXT: lit.ref.store %a, %a_2

    # CHECK-NEXT: lit.ref.store %b, %a_2
    a = b  # Arguments are mutable!

    # CHECK-NEXT: lit.ref.store %b, %a_2
    a = b  # Subsequent arguments don't re-make the box.

    # CHECK-NEXT: lit.call {{.*}}MemoryOnly::@"__init__{{.*}}(%mem_1)
    mem = MemoryOnly()

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}NonTrivialReg::@"__init__{{.*}}()
    # CHECK-NEXT: lit.ref.store [[TMP]], %reg_0
    reg = NonTrivialReg()

    # Issue#38762
    # MOCO-83: [mojo][Bug] def methods can't shadow names via assignment
    defTests = 4


struct TypeWithParametricSelf:
    fn method(ref self):
        pass


struct ValueWithTypeWithParametricSelf:
    var member: TypeWithParametricSelf


# CHECK-LABEL: test_def_arg_box_mbvalue
def test_def_arg_box_mbvalue(
    a: TypeWithParametricSelf, b: ValueWithTypeWithParametricSelf
):
    # CHECK-NEXT: %xyz = lit.var.decl "xyz"
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}method{{.*}}(%a)
    # CHECK-NEXT: lit.ref.store [[TMP]], %xyz
    var xyz = a.method()

    # MOCO-715: failed to infer implicit parameter 'is_mutable' of argument 'self' type 'Pointer
    # CHECK-NEXT: [[MEMBERREF:%.*]] = lit.ref.struct.ger %b[member]
    _ = b.member.method()

fn use(a: Int): pass
fn use(a: String): pass

# https://github.com/modular/mojo/issues/3955
# Unexpected copy-on-write behaviour with for loops
# CHECK-LABEL: test_mutable_def_arg_emission
def test_mutable_def_arg_emission(byte: Int, str: String):
   # CHECK-NEXT: %str_0 = lit.var.decl "str"
   # CHECK-NEXT: lit.call {{.*}}String::@"__copyinit__{{.*}}(%str, %str_0)
   # CHECK-NEXT: %byte_1 = lit.var.decl "byte"
   # CHECK-NEXT: lit.ref.store %byte, %byte_1

   # CHECK: } body {
    while True:
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %byte_1
        # CHECK-NEXT: lit.call {{.*}}use(Int)"([[TMP]])
        use(byte)

        # CHECK: lit.call {{.*}}@Int::@"__iadd__{{.*}}(%byte_1,
        byte += 1
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %byte_1
        # CHECK-NEXT: lit.call {{.*}}use(Int)"([[TMP]])
        use(byte)
        # CHECK-NEXT: lit.break
        break

    # CHECK: } body {
    while True:
        # CHECK-NEXT: [[TMP:%.*]] = kgen.rebind %str_0
        # CHECK-NEXT: lit.call {{.*}}use(String)"{{.*}}([[TMP]])
        use(str)
        # CHECK: lit.call {{.*}}String::@"__iadd__{{.*}}(%str_0,
        str += ""
        # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut %str_0
        # CHECK: lit.call {{.*}}use(String)"{{.*}}([[TMP]])
        use(str)
        break

fn returnsMultiple() -> (Int, MemoryOnly):
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
