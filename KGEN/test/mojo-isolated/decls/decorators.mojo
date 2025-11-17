# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

from builtin.stubs import _get_kgen_string

# ===----------------------------------------------------------------------=== #
# Function decorators
# ===----------------------------------------------------------------------=== #

struct NoDebugInlineTest:
    # Two decorators stacked up
    @always_inline("nodebug")
    @staticmethod
    fn test():
        return


# Test some graph compiler decorators.
fn elementwise(): return

fn register(a: StringLiteral): return

# CHECK-LABEL: lit.fn @"decorated_fn()"
# CHECK-NEXT: decorators <:!lit.generator<() -> !kgen.none> @{{.*}}::@"elementwise()">
@elementwise
fn decorated_fn():
    pass

# CHECK-LABEL: lit.struct.decl @DecoratedStruct
# CHECK: decorators <:none apply({{.*}}register{{.*}}<:string "hello">
@register("hello")
struct DecoratedStruct:
    pass

# ===----------------------------------------------------------------------=== #
# @always_inline
# ===----------------------------------------------------------------------=== #

# CHECK: lit.fn @"test_always_inline()"() -> index always_inline
@always_inline
fn test_always_inline() -> __mlir_type.index:
    return Int(1)._mlir_value

# CHECK-LABEL: lit.fn @"test_always_inline_no_debug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
fn test_always_inline_no_debug():
    pass


# CHECK-LABEL: lit.fn @"math{{.*}} always_inline_builtin
@always_inline("builtin")
fn math(a: __mlir_type.index, b: __mlir_type.index) -> __mlir_type.index:
    return __mlir_op.`index.add`(a, b)

# CHECK-LABEL: lit.fn @"use_math
fn use_math(a: __mlir_type.index) -> __mlir_type.index:
    # CHECK: %index = kgen.param.constant = <{{.*}}, 1, 2), 3)>
    # CHECK: %0 = lit.call @decorators::@"math(
    # CHECK: lit.return %0 : index
    return math(
        a,
        math(
            __mlir_op.`index.constant`[value = __mlir_attr.`1:index`](),
            __mlir_op.`index.constant`[value = __mlir_attr.`2:index`](),
        ),
    )


# https://github.com/modularml/modular/issues/8500
struct AlwaysInlineByRef:
    @always_inline("nodebug")
    fn do_by_ref(mut self):
        pass

fn test_inline_by_ref(mut a: AlwaysInlineByRef):
    a.do_by_ref()

# ===----------------------------------------------------------------------=== #
# @staticmethod
# ===----------------------------------------------------------------------=== #

# TODO

# ===----------------------------------------------------------------------=== #
# @no_inline
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"test_no_inline
# CHECK-SAME: no_inline
@no_inline
fn test_no_inline():
    pass

##===----------------------------------------------------------------------===##
# @deprecated
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.struct.decl @DeprecatedStruct
# CHECK-SAME: deprecationWarning = "struct"
@deprecated("struct")
struct DeprecatedStruct:
    pass


# CHECK-LABEL: lit.fn @"deprecated_func
# CHECK-SAME: deprecationWarning = "func"
@deprecated("func")
fn deprecated_func():
    pass


# CHECK-LABEL: lit.trait.decl @DeprecatedTrait
# CHECK-SAME: deprecationWarning = "trait"
@deprecated("trait")
trait DeprecatedTrait:
    pass


# CHECK-LABEL: lit.alias.decl *"deprecated_alias
# CHECK-SAME: deprecationWarning = "alias"
@deprecated("alias")
comptime deprecated_alias = 1


##===----------------------------------------------------------------------===##
# @deprecated(use)
##===----------------------------------------------------------------------===##

struct DeprecatedStructTarget:
    pass

# CHECK-LABEL: lit.struct.decl @DeprecatedStructUse
# CHECK-SAME: deprecationWarning = "'DeprecatedStructUse' is deprecated, use 'DeprecatedStructTarget' instead"
@deprecated(use=DeprecatedStructTarget)
struct DeprecatedStructUse:
    pass

fn deprecated_func_target():
    pass

# CHECK-LABEL: lit.fn @"deprecated_func_use
# CHECK-SAME: deprecationWarning = "'deprecated_func_use' is deprecated, use 'deprecated_func_target' instead"
@deprecated(use=deprecated_func_target)
fn deprecated_func_use():
    pass

trait DeprecatedTraitTarget:
    pass

# CHECK-LABEL: lit.trait.decl @DeprecatedTraitUse
# CHECK-SAME: deprecationWarning = "'DeprecatedTraitUse' is deprecated, use 'DeprecatedTraitTarget' instead"
@deprecated(use=DeprecatedTraitTarget)
trait DeprecatedTraitUse:
    pass

comptime deprecated_alias_target = 1

# CHECK-LABEL: lit.alias.decl *"deprecated_alias_use
# CHECK-SAME: deprecationWarning = "'deprecated_alias_use' is deprecated, use 'deprecated_alias_target' instead"
@deprecated(use=deprecated_alias_target)
comptime deprecated_alias_use = 1

# ===----------------------------------------------------------------------=== #
# @implicit
# ===----------------------------------------------------------------------=== #

struct DeprecatedImplicitConversion:
    @implicit(deprecated=True)
    fn __init__(out self, value: Int):
        pass

struct NotDeprecatedImplicitConversion:
    @implicit(deprecated=False)
    fn __init__(out self, value: Int):
        pass

fn foo(y: DeprecatedImplicitConversion): pass

fn foo(z: Int): pass

fn deprecated_implicit_conversion():
    # There should be no warnings here.
    _: NotDeprecatedImplicitConversion = 1
    _ = DeprecatedImplicitConversion(1)

    # There should be no warning here because the `Int` overload is selected.
    foo(Int(1))

# ===----------------------------------------------------------------------=== #
# @__llvm_metadata
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"kernel{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = ["nvvm.maxntid", {{.*}}#pop.array<x> : !pop.array<
@__llvm_metadata(
    `nvvm.maxntid`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel[x: Int]():
    pass

# CHECK-LABEL: lit.fn @"kernel_1{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
comptime mname = "nvvm.maxntid"

@__llvm_metadata(
    mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel_1[x: Int]():
    pass

# CHECK-LABEL: lit.fn @"kernel_2{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
@__llvm_metadata(
    `mname`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel_2[x: Int]():
    pass

# TODO: Figure out how to get the value of the alias.
# CHECK-LABEL: lit.fn @"kernel_3{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [
# CHECK-SAME: data_to_str({{.*}}alias_parametric_fn
# CHECK-SAME: #kgen.unknown : !lit.struct<#IntLiteral <:!pop.int_literal 128>>
fn alias_parametric_fn() -> StaticString:
    @parameter
    if True:
        return "nvvm.maxntid"
    else:
        return "rocdl.flat_work_group_size"


comptime mname1 = _get_kgen_string[alias_parametric_fn()]()

@__llvm_metadata(mname1=128)
fn kernel_3[x: Int]():
    pass

# ===----------------------------------------------------------------------=== #
# @__llvm_arg_metadata
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"llvm_arg_meta
# CHECK-SAME{LITERAL}: LLVMArgMetadataArray = [[], ["nvvm.grid_constant", unit, "myMeta", unit], [], [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, #pop.array<x> : !pop.array<1, !Int>], []]
@__llvm_arg_metadata(b, `nvvm.grid_constant`, `myMeta`)
@__llvm_arg_metadata(c)
@__llvm_arg_metadata(
    d, mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn llvm_arg_meta[x: Int](a: Int, b: Int, c: Int, d: Int, e: Int):
    pass


# ===----------------------------------------------------------------------=== #
# Struct decorators
# ===----------------------------------------------------------------------=== #

##===----------------------------------------------------------------------===##
# Struct @fieldwise_init decorator
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.struct.decl @StructExample
@register_passable
struct StructExample(ImplicitlyCopyable):
    fn __copyinit__(out self, other: Self):
        pass

    fn __init__(out self):
        pass


# CHECK-LABEL: lit.struct.decl @ValueMem(!AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility)
# CHECK: move :!lit.generator<[2]({{.*}} owned_in_mem, |, ?, {{.*}} byref_result) {{.*}}ValueMem::@"__moveinit__
@fieldwise_init
struct ValueMem(ImplicitlyCopyable, Movable):
    var a: Int  # Trivial
    var b: StructExample  # Copy ctor


# CHECK: lit.fn @"__moveinit__(
# CHECK-SAME:  %other: !lit.ref<!ValueMem, mut {{.*}}> owned_in_mem,
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> byref_result)
# CHECK-SAME: -> !kgen.none always_inline_no_debug attributes
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %other[a]
# CHECK-NEXT: %2 = lit.load.consume %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %other[b]
# CHECK-NEXT: %5 = lit.load.consume %4
# CHECK-NEXT: lit.ref.store %5, %3

# CHECK: lit.fn @"__copyinit__(
# CHECK-SAME:  %other: !lit.ref<!ValueMem, imm {{.*}}> read_mem,
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> byref_result)
# CHECK-SAME: -> !kgen.none always_inline_no_debug attributes
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %other[a]
# CHECK-NEXT: %2 = lit.ref.load %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %other[b]
# CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%4)
# CHECK-NEXT: lit.ref.store [[TMP]], %3

# CHECK: lit.fn @"__init__(
# CHECK-SAME:  %a: !Int,
# CHECK-SAME:  %b: !lit.ref<!StructExample, mut *"b`"> owned_in_mem,
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> byref_result
# CHECK-SAME: ) -> !kgen.none always_inline_no_debug attributes {isStatic, sourceName = "__init__", specialFnKind = 2 : i8, synthetic} {
# CHECK-NEXT: %[[PA:.*]] = lit.ref.struct.ger %self[a]
# CHECK-NEXT: lit.ref.store %a, %[[PA]]
# CHECK-NEXT: %[[PB:.*]] = lit.ref.struct.ger %self[b]
# CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %b
# CHECK-NEXT: lit.ref.store [[TMP]], %[[PB]]


# CHECK-LABEL: lit.struct.decl @ValueMemHasCopy(!AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility)
@fieldwise_init
struct ValueMemHasCopy(ImplicitlyCopyable, Movable):
    var a: Int
    var b: StructExample

    fn __copyinit__(out self, other: Self):
        self.a = other.a
        self.b = other.b


# CHECK-LABEL: lit.struct.decl @ValueMemHasMove(!AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility)
@fieldwise_init
struct ValueMemHasMove(Movable, ImplicitlyCopyable):
    var a: Int
    var b: StructExample


# CHECK-LABEL: lit.struct.decl @ValueRegTrivial
# CHECK-SAME: (!AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility) register_passable_trivial

# CHECK: lit.fn @"__moveinit__{{.*}}"[{{.*}}](%other: !lit.ref<!ValueRegTrivial, {{.*}}> owned_in_mem,
# CHECK-SAME: %self: !lit.ref<!ValueRegTrivial, {{.*}}> byref_result)
# CHECK-NEXT: [[V0:%.*]] = lit.ref.load %other : <!ValueRegTrivial
# CHECK-NEXT: lit.ref.store [[V0]], %self
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.ownership.mark_destroyed %other
# CHECK-NEXT: lit.return %none : !kgen.none

# CHECK: lit.fn @"__copyinit__{{.*}}"[{{.*}}](%other: !lit.ref<!ValueRegTrivial, {{.*}}> read_mem,
# CHECK-SAME: %self: !lit.ref<!ValueRegTrivial, {{.*}}> byref_result) -> !kgen.none always_inline_no_debug
# CHECK-NEXT: [[V0:%.*]] = lit.ref.load %other : <!ValueRegTrivial
# CHECK-NEXT: lit.ref.store [[V0]], %self
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %none : !kgen.none

@fieldwise_init
@register_passable("trivial")
struct ValueRegTrivial(Copyable):
    var a: __mlir_type.index


# CHECK-LABEL: lit.struct.decl @ValueReg
@fieldwise_init
@register_passable
struct ValueReg(ImplicitlyCopyable):
    var a: Int
    var b: StructExample


# CHECK: lit.fn @"__copyinit__
# CHECK-SAME: (%other: !lit.ref<!ValueReg, imm *"existing`"> read_mem,
# CHECK-SAME : %self: !lit.ref<!ValueReg, mut *"self`"> byref_result)
# CHECK-SAME: attributes {{.*}}specialFnKind = 3 : i8
# CHECK-NEXT: [[SELFA:%.*]] = lit.ref.struct.ger %self[a]
# CHECK-NEXT: [[OTHERA:%.*]] = lit.ref.struct.ger %other[a]
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.load [[OTHERA]]
# CHECK-NEXT: lit.ref.store [[TMP]], [[SELFA]]
# CHECK-NEXT: [[SELFB:%.*]] = lit.ref.struct.ger %self[b]
# CHECK-NEXT: [[OTHERB:%.*]] = lit.ref.struct.ger %other[b]
# CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[OTHERB]])
# CHECK-NEXT: lit.ref.store [[TMP]], [[SELFB]]

# CHECK: lit.fn @"__init__(
# CHECK-SAME:  (
# CHECK-SAME:  %a: !Int,
# CHECK-SAME:  %b: !lit.ref<!StructExample, mut *"b`"> owned_in_mem
# CHECK-SAME: ) -> !ValueReg
# CHECK-NEXT: %self = lit.var.decl "self"
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: lit.ref.store %a, %0
# CHECK-NEXT: %1 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %2 =  lit.load.consume %b
# CHECK-NEXT: lit.ref.store %2, %1
# CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %self
# CHECK-NEXT: lit.return [[TMP]]


# COM: Ensure that "self" is a valid field name.
# CHECK-LABEL: lit.struct.decl @Foo(!AnyType_Copyable_ImplicitlyCopyable_Movable_UnknownDestructibility) attributes
@fieldwise_init
struct Foo(ImplicitlyCopyable, Movable):
    var a: Int
    var self: Int


# CHECK: lit.fn @"__init__{{.*}}(%a: !Int, %self: !Int, ?, %self_0[self]: !lit.ref<!Foo, mut {{.*}}> byref_result)


# CHECK-LABEL: lit.struct.decl @ParamVarArg<I: variadic<!Int> pos_vararg>
@fieldwise_init
@register_passable("trivial")
struct ParamVarArg[*I: Int]:
    pass


# CHECK-LABEL: lit.struct.decl @TraitMember
@fieldwise_init
struct TraitMember[T: ImplicitlyCopyable](ImplicitlyCopyable, Movable):
    var value: T
    # CHECK: lit.fn @"__moveinit__
    # CHECK: call{{.*}}__copyinit__
    # CHECK: lit.fn @"__copyinit__
    # CHECK: call{{.*}}__copyinit__


# CHECK: lit.fn @"notSynthetic{{.*}}(%self: !lit.ref<!NotSynthetic, imm {{.*}}> read_mem) -> !kgen.none attributes {sourceName = "notSynthetic", specialFnKind = 0 : i8}
# CHECK: lit.fn @"__moveinit__{{.*}}synthetic
# CHECK: lit.fn @"__copyinit__{{.*}}synthetic
# CHECK: lit.fn @"__init__{{.*}}synthetic
@fieldwise_init
struct NotSynthetic(ImplicitlyCopyable, Movable):
    var member: __mlir_type.`index`

    fn notSynthetic(self):
        pass


# CHECK-LABEL: lit.struct.decl @VarArgInit
@fieldwise_init
@register_passable("trivial")
struct VarArgInit:
    var a: Int

    # CHECK: lit.fn @"__init__(decorators::ValueMem*)"{{.*}}(%values: !kgen.variadic<!lit.ref<!ValueMem, imm {{.*}}>> read_mem|pos_vararg
    # The argument is intentionally memory-only.
    @implicit
    fn __init__(out self, *values: ValueMem):
        self.a = 42

    # CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !VarArgInit


# COM: Body resolution of `Node` will recurse on itself. Make sure that the
# COM: trait requirements for ImplicitlyCopyable and Movable are generated early.
struct BoxCopyable[T: ImplicitlyCopyable]:
    pass


@fieldwise_init
struct Node(ImplicitlyCopyable, Movable):
    var id: RecursiveCopyable.ID


# CHECK-LABEL: lit.struct.decl @RecursiveCopyable
struct RecursiveCopyable:
    comptime ID = Int
    # CHECK: lit.struct.field recurse
    # CHECK-SAME: <:!ImplicitlyCopyable !Node>
    var recurse: BoxCopyable[Node]


# CHECK-LABEL: lit.struct.decl @RaisingFieldwiseInit
struct RaisingFieldwiseInit(ImplicitlyCopyable, Movable):
    var x: Int

    # CHECK-LABEL: lit.fn @"__init__{{.*}} throws
    fn __init__(out self, x: Int) raises:
        pass

fn register_internal(x: StaticString):
    pass

# CHECK-LABEL: lit.struct.decl @DecoratorOrder1
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationWarning = "DecoratorOrder1"
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder1
@register_internal("custom.op")
@deprecated("DecoratorOrder1")
@fieldwise_init
@register_passable("trivial")
struct DecoratorOrder1:
    var a: Int

# CHECK-LABEL: lit.struct.decl @DecoratorOrder2
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationWarning = "DecoratorOrder2"
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder2
@deprecated("DecoratorOrder2")
@register_internal("custom.op")
@register_passable("trivial")
@fieldwise_init
struct DecoratorOrder2:
    var a: Int

# CHECK-LABEL: lit.struct.decl @DecoratorOrder3
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationWarning = "DecoratorOrder3"
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder3
@register_passable("trivial")
@fieldwise_init
@deprecated("DecoratorOrder3")
@register_internal("custom.op")
struct DecoratorOrder3:
    var a: Int

# CHECK-LABEL: lit.struct.decl @DecoratorOrder4
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationWarning = "DecoratorOrder4"
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder4
@fieldwise_init
@register_passable("trivial")
@register_internal("custom.op")
@deprecated("DecoratorOrder4")
struct DecoratorOrder4:
    var a: Int
