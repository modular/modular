# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

from std.builtin.stubs import _get_kgen_string

# ===----------------------------------------------------------------------=== #
# Function decorators
# ===----------------------------------------------------------------------=== #

struct NoDebugInlineTest:
    # Two decorators stacked up
    @always_inline("nodebug")
    @staticmethod
    def test():
        return


# Test some graph compiler decorators.
def elementwise(): return

def register(a: StringLiteral): return

# CHECK-LABEL: lit.fn @"decorated_fn()"
# CHECK-NEXT: decorators <:!lit.generator<() -> !kgen.none> @{{.*}}::@"elementwise()">
@elementwise
def decorated_fn():
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
def test_always_inline() -> __mlir_type.index:
    return Int(1)._mlir_value

# CHECK-LABEL: lit.fn @"test_always_inline_no_debug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
def test_always_inline_no_debug():
    pass


# CHECK-LABEL: lit.fn @"math{{.*}} always_inline_builtin
@always_inline("builtin")
def math(a: __mlir_type.index, b: __mlir_type.index) -> __mlir_type.index:
    return __mlir_op.`index.add`(a, b)

# CHECK-LABEL: lit.fn @"use_math
def use_math(a: __mlir_type.index) -> __mlir_type.index:
    # CHECK: %index3 = kgen.param.constant = <3>
    # CHECK: %0 = lit.call tail @decorators::@"math(
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
    def do_by_ref(mut self):
        pass

def test_inline_by_ref(mut a: AlwaysInlineByRef):
    a.do_by_ref()


struct AIBuiltinPair(TrivialRegisterPassable):
    var a: Int
    var b: Int

    comptime MyAlias = AIBuiltinPair

    @always_inline("builtin")
    def __init__(out self: Self.MyAlias, x: Int, y: Int):
        self.a = x
        self.b = y

# CHECK-LABEL: lit.fn @"test_ai_builtin_pair
def test_ai_builtin_pair():
    # CHECK-NEXT: lit.alias.decl *"example{{.*}}!AIBuiltinPair {a: !Int = {1}, b: !Int = {2}}
    comptime example = AIBuiltinPair(1, 2)

# ===----------------------------------------------------------------------=== #
# @staticmethod
# ===----------------------------------------------------------------------=== #


struct StaticMethodTest:
    # CHECK-LABEL: lit.fn @"some_static_method()"()
    # CHECK-SAME: isStatic
    @staticmethod
    def some_static_method():
        pass


# ===----------------------------------------------------------------------=== #
# @no_inline
# ===----------------------------------------------------------------------=== #

# CHECK-LABEL: lit.fn @"test_no_inline
# CHECK-SAME: no_inline
@no_inline
def test_no_inline():
    pass

# ===----------------------------------------------------------------------=== #
# @implicit
# ===----------------------------------------------------------------------=== #

struct DeprecatedImplicitConversion:
    @implicit(deprecated=True)
    def __init__(out self, value: Int):
        pass

struct NotDeprecatedImplicitConversion:
    @implicit(deprecated=False)
    def __init__(out self, value: Int):
        pass

def foo(y: DeprecatedImplicitConversion): pass

def foo(z: Int): pass

def deprecated_implicit_conversion():
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
def kernel[x: Int]():
    pass

# CHECK-LABEL: lit.fn @"kernel_1{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
comptime mname = "nvvm.maxntid"

@__llvm_metadata(
    mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
def kernel_1[x: Int]():
    pass

# CHECK-LABEL: lit.fn @"kernel_2{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
@__llvm_metadata(
    `mname`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
def kernel_2[x: Int]():
    pass

# TODO: Figure out how to get the value of the alias.
# CHECK-LABEL: lit.fn @"kernel_3{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [
# CHECK-SAME: data_to_str({{.*}}alias_parametric_fn
# CHECK-SAME: #kgen.unknown : !lit.struct<#IntLiteral <:!pop.int_literal 128>>
def alias_parametric_fn() -> StaticString:
    comptime if True:
        return "nvvm.maxntid"
    else:
        return "rocdl.flat_work_group_size"


comptime mname1 = _get_kgen_string[alias_parametric_fn()]()

@__llvm_metadata(mname1=128)
def kernel_3[x: Int]():
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
def llvm_arg_meta[x: Int](a: Int, b: Int, c: Int, d: Int, e: Int):
    pass


# ===----------------------------------------------------------------------=== #
# Struct decorators
# ===----------------------------------------------------------------------=== #

def register_internal(x: StaticString):
    pass

# CHECK-LABEL: lit.struct.decl @DecoratorOrder1
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationInfo = #lit.deprecation<"DecoratorOrder1">
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder1
@register_internal("custom.op")
@deprecated("DecoratorOrder1")
@fieldwise_init
struct DecoratorOrder1(TrivialRegisterPassable):
    var a: Int

# CHECK-LABEL: lit.struct.decl @DecoratorOrder2
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationInfo = #lit.deprecation<"DecoratorOrder2">
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder2
@deprecated("DecoratorOrder2")
@register_internal("custom.op")
@fieldwise_init
struct DecoratorOrder2(TrivialRegisterPassable):
    var a: Int

# CHECK-LABEL: lit.struct.decl @DecoratorOrder3
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationInfo = #lit.deprecation<"DecoratorOrder3">
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder3
@fieldwise_init
@deprecated("DecoratorOrder3")
@register_internal("custom.op")
struct DecoratorOrder3(TrivialRegisterPassable):
    var a: Int

# CHECK-LABEL: lit.struct.decl @DecoratorOrder4
# CHECK-SAME: register_passable_trivial
# CHECK-SAME: deprecationInfo = #lit.deprecation<"DecoratorOrder4">
# CHECK: decorators <{{.*}}:string "custom.op"
# CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !DecoratorOrder4
@fieldwise_init
@register_internal("custom.op")
@deprecated("DecoratorOrder4")
struct DecoratorOrder4(TrivialRegisterPassable):
    var a: Int


##===----------------------------------------------------------------------===##
# Struct @fieldwise_init decorator
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.struct.decl @StructExample
struct StructExample(ImplicitlyCopyable, RegisterPassable):
    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self):
        pass


# CHECK-LABEL: lit.struct.decl @ValueMem(!AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDestructible_Movable)
# CHECK: move :!lit.generator<[2]({{.*}} deinit_mem, ?, {{.*}} byref_result) {{.*}}@ValueMem::@"__init__(take:
@fieldwise_init
struct ValueMem(ImplicitlyCopyable):
    var a: Int  # Trivial
    var b: StructExample  # Copy ctor


# CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*,
# CHECK-SAME:  %take: !lit.ref<!ValueMem, mut {{.*}}> deinit_mem,
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> byref_result)
# CHECK-SAME: -> !kgen.none always_inline_no_debug attributes
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %take[a]
# CHECK-NEXT: %2 = lit.load.consume %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %take[b]
# CHECK-NEXT: %5 = lit.load.consume %4
# CHECK-NEXT: lit.ref.store %5, %3

# CHECK: lit.fn @"__init__{{.*}}"{{.*}}*,
# CHECK-SAME:  %copy: !lit.ref<!ValueMem, imm {{.*}}> read_mem,
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> byref_result)
# CHECK-SAME: -> !kgen.none always_inline_no_debug attributes
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %copy[a]
# CHECK-NEXT: %2 = lit.ref.load %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %copy[b]
# CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}"{{.*}}(%4){{.*}}*, "copy"
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


# CHECK-LABEL: lit.struct.decl @ValueMemHasCopy(!AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDestructible_Movable)
@fieldwise_init
struct ValueMemHasCopy(ImplicitlyCopyable):
    var a: Int
    var b: StructExample


# CHECK-LABEL: lit.struct.decl @ValueMemHasMove(!AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDestructible_Movable)
@fieldwise_init
struct ValueMemHasMove(Movable, ImplicitlyCopyable):
    var a: Int
    var b: StructExample


# CHECK-LABEL: lit.struct.decl @ValueRegTrivial
# CHECK-SAME: (!AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDestructible_Movable_RegisterPassable_TrivialRegisterPassable) register_passable_trivial

# CHECK: lit.fn @"__init__{{.*}}"{{.*}}[{{.*}}](*, %take: !lit.ref<!ValueRegTrivial, {{.*}}> deinit_mem,
# CHECK-SAME: %self: !lit.ref<!ValueRegTrivial, {{.*}}> byref_result)
# CHECK-NEXT: [[V0:%.*]] = lit.ref.load %take : <!ValueRegTrivial
# CHECK-NEXT: lit.ref.store [[V0]], %self
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.ownership.mark_destroyed %take
# CHECK-NEXT: lit.return %none : !kgen.none

# CHECK: lit.fn @"__init__{{.*}}"{{.*}}[{{.*}}](*, %copy: !lit.ref<!ValueRegTrivial, {{.*}}> read_mem,
# CHECK-SAME: %self: !lit.ref<!ValueRegTrivial, {{.*}}> byref_result) -> !kgen.none always_inline_no_debug
# CHECK-NEXT: [[V0:%.*]] = lit.ref.load %copy : <!ValueRegTrivial
# CHECK-NEXT: lit.ref.store [[V0]], %self
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: lit.return %none : !kgen.none

@fieldwise_init
struct ValueRegTrivial(TrivialRegisterPassable, Copyable):
    var a: __mlir_type.index


# CHECK-LABEL: lit.struct.decl @ValueReg
@fieldwise_init
struct ValueReg(ImplicitlyCopyable, RegisterPassable):
    var a: Int
    var b: StructExample


# CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %copy: !lit.ref<!ValueReg, imm *"copy`"> read_mem,
# CHECK-SAME : %self: !lit.ref<!ValueReg, mut *"self`"> byref_result)
# CHECK-SAME: attributes {{.*}}specialFnKind = 3 : i8
# CHECK-NEXT: [[SELFA:%.*]] = lit.ref.struct.ger %self[a]
# CHECK-NEXT: [[OTHERA:%.*]] = lit.ref.struct.ger %copy[a]
# CHECK-NEXT: [[TMP:%.*]] = lit.ref.load [[OTHERA]]
# CHECK-NEXT: lit.ref.store [[TMP]], [[SELFA]]
# CHECK-NEXT: [[SELFB:%.*]] = lit.ref.struct.ger %self[b]
# CHECK-NEXT: [[OTHERB:%.*]] = lit.ref.struct.ger %copy[b]
# CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}"{{.*}}([[OTHERB]]){{.*}}*, "copy"
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
# CHECK-LABEL: lit.struct.decl @Foo(!AnyType_Copyable_ImplicitlyCopyable_ImplicitlyDestructible_Movable) attributes
@fieldwise_init
struct Foo(ImplicitlyCopyable):
    var a: Int
    var self: Int


# CHECK: lit.fn @"__init__{{.*}}(%a: !Int, %self: !Int, ?, %self_0[self]: !lit.ref<!Foo, mut {{.*}}> byref_result)


# CHECK-LABEL: lit.struct.decl @ParamVarArg<I: variadic<!Int> pos_vararg>
@fieldwise_init
struct ParamVarArg[*I: Int](TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.struct.decl @TraitMember
@fieldwise_init
struct TraitMember[T: ImplicitlyCopyable](ImplicitlyCopyable):
    var value: Self.T
    # CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %take:
    # CHECK: lit.call{{.*}}__init__(take:$0$)">
    # CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %copy:
    # CHECK: lit.call{{.*}}__init__(copy:$0)">


# CHECK: lit.fn @"notSynthetic{{.*}}(%self: !lit.ref<!NotSynthetic, imm {{.*}}> read_mem) -> !kgen.none attributes {sourceName = "notSynthetic", specialFnKind = 0 : i8}
# CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %take:{{.*}}synthetic
# CHECK: lit.fn @"__init__{{.*}}"{{.*}}(*, %copy:{{.*}}synthetic
# CHECK: lit.fn @"__init__{{.*}}synthetic
@fieldwise_init
struct NotSynthetic(ImplicitlyCopyable):
    var member: __mlir_type.`index`

    def notSynthetic(self):
        pass


# CHECK-LABEL: lit.struct.decl @VarArgInit
@fieldwise_init
struct VarArgInit(TrivialRegisterPassable):
    var a: Int

    # CHECK: lit.fn @"__init__{{.*}}(decorators::ValueMem*)"{{.*}}(%values: {{.*}}> read_mem|pos_vararg
    # The argument is intentionally memory-only.
    @implicit
    def __init__(out self, *values: ValueMem):
        self.a = 42

    # CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !VarArgInit


# COM: Body resolution of `Node` will recurse on itself. Make sure that the
# COM: trait requirements for ImplicitlyCopyable and Movable are generated early.
struct BoxCopyable[T: ImplicitlyCopyable]:
    pass


@fieldwise_init
struct Node(ImplicitlyCopyable):
    var id: RecursiveCopyable.ID


# CHECK-LABEL: lit.struct.decl @RecursiveCopyable
struct RecursiveCopyable:
    comptime ID = Int
    # CHECK: lit.struct.field recurse
    # CHECK-SAME: <:!ImplicitlyCopyable !Node>
    var recurse: BoxCopyable[Node]


# CHECK-LABEL: lit.struct.decl @RaisingFieldwiseInit
struct RaisingFieldwiseInit(ImplicitlyCopyable):
    var x: Int

    # CHECK-LABEL: lit.fn @"__init__{{.*}} throws
    def __init__(out self, x: Int) raises:
        pass
