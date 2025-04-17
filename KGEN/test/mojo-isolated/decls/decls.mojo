# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s

from builtin.stubs import _get_kgen_string

##===----------------------------------------------------------------------===##
# fn/def
##===----------------------------------------------------------------------===##


# Method overloading.
# CHECK-LABEL: lit.fn @"testThing({{.*}}Int)"
fn testThing(a: Int) -> FloatDyn:
    return 1.0


# CHECK-LABEL: lit.fn @"testThing({{.*}}Int,{{.*}}Int)"
fn testThing(a: Int, b: Int) -> Int:
    return 1


fn implicit_variable_decls(a: Int) -> Int:
    b = a + a
    return b


alias IntToFloat32Type = fn (Int) -> FloatDyn


fn takeIntToFloat32Param[f: IntToFloat32Type]():
    pass


fn varargOverload(a: Int):
    pass


fn varargOverload(*a: Int):
    pass


fn varargOverload():
    pass


fn packOverload(a: Int):
    pass


fn packOverload[*Ts: AnyType](*a: *Ts):
    pass


fn packOverload():
    pass


fn directly_pass_pack(pack: __mlir_type.`!kgen.pack<[index]>`):
    pass


# Varargs + traits are a thing.
# https://github.com/modular/mojo/issues/1443
fn variadic_trait_elt[T: Copyable](*xs: T):
    pass


# CHECK-LABEL: lit.fn @"trait_pack
# CHECK-SAME: <{{.*}}, Ts:
# CHECK-SAME: %rest: !lit.ref<@stdlib::@builtin::@stubs::@VariadicPack<:!Bool {:i1 0}, {{.*}}origin<0> = *"rest`1"}, :!lit.anytrait<!AnyType> !Copyable, :variadic<!Copyable> Ts>, imm *"rest`2"> read_mem|pack)
fn trait_pack[T: Copyable, *Ts: Copyable](first: T, *rest: *Ts):
    pass


# CHECK-LABEL: lit.fn @"callOverload
fn callOverload(a: Int, pack: __mlir_type.`!kgen.pack<[index]>`):
    # CHECK: lit.call @decls::@"testThing({{.*}}Int)"(%a)
    _ = testThing(a)
    # CHECK: lit.call @decls::@"testThing({{.*}}Int,{{.*}}Int)"(%a, %a)
    _ = testThing(a, a)

    # CHECK: kgen.create_closure[!lit.generator<(!Int, |) -> !FloatDyn>:
    # CHECK-SAME: rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing({{.*}}Int)")]()
    var float1: IntToFloat32Type = testThing

    # CHECK: kgen.create_closure[!lit.generator<(!Int, |) -> !FloatDyn>:
    # CHECK-SAME: rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing({{.*}}Int)")]()
    # CHECK-NEXT: lit.ref.store %3, %float1
    float1 = testThing

    # CHECK: %4 = kgen.create_closure[!lit.generator<(!Int, |) -> !FloatDyn>:
    # CHECK-SAME: rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing({{.*}}Int)")]()
    var float2: IntToFloat32Type = testThing

    # CHECK: lit.call @decls::@"takeIntToFloat32Param[fn({{.*}}Int, /) -> {{.*}}FloatDyn]()"<:
    # CHECK-SAME: !lit.generator<(!Int, |) -> !FloatDyn> rebind(:!lit.generator<("a": !Int) -> !FloatDyn> @decls::@"testThing{{.*}}")>()
    takeIntToFloat32Param[testThing]()

    # Issue #10036.  This should call the exact match, consider the varargs match
    # less specific.
    # CHECK: lit.call @decls::@"varargOverload({{.*}}Int)"(%{{.*}})
    varargOverload(2)

    # CHECK:  lit.call @decls::@"varargOverload()"()
    varargOverload()

    # Expect packs to behave similarly to varargs.
    # CHECK: %[[IDX3:.*]] = {{.*}}constant{{.*}}3
    # CHECK: lit.call @decls::@"packOverload({{.*}}Int)"(%[[IDX3]])
    packOverload(3)
    # CHECK:  lit.call @decls::@"packOverload()"()
    packOverload()

    # CHECK-NOT: pack.create
    # CHECK: call {{.*}}directly_pass_pack{{.*}}(%pack)
    directly_pass_pack(pack)

    # CHECK: call {{.*}}trait_pack
    # CHECK-SAME: [!Int, {"__copyinit__"
    trait_pack(1, 2, 3)


@register_passable("trivial")
struct MyInt:
    var value: Int

    @implicit
    @always_inline("nodebug")
    @implicit
    fn __init__(out self, _a: Int):
        self.value = _a


fn paramOverload[x: Int]():
    pass


fn paramOverload[x: Int, y: Int]():
    pass


fn paramOverload[*x: Int]():
    pass


fn paramOverload(y: Int):
    pass


fn paramOverload[x: Int, T: AnyTrivialRegType](y: T):
    pass


fn paramOverload[*x: Int](y: Int):
    pass


fn paramOverload2[*x: Int]():
    pass


fn paramOverload2[x: MyInt]():
    pass


fn paramOverload2[x: MyInt, y: MyInt]():
    pass


fn paramOverload2[*x: MyInt]():
    pass


# CHECK-LABEL: lit.fn @"callParametricOverload
fn callParametricOverload[a: Int, b: Int, c: Int](x: Int):
    # CHECK-NEXT: lit.call @decls::@"paramOverload[{{.*}}Int]()"
    paramOverload[a]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload{{.*}}<:!Int a, :!Int b>()
    paramOverload[a, b]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload{{.*}}<:variadic<!Int> [a, b, c]>()
    paramOverload[a, b, c]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload({{.*}}Int)"
    paramOverload(x)

    # CHECK-NEXT: lit.call @decls::@"paramOverload[{{.*}}Int,AnyTrivialRegType]($1)"
    paramOverload[a](x)

    # CHECK-NEXT: lit.call @decls::@"paramOverload{{.*}}<:variadic<!Int> [a, b]>(%x)
    paramOverload[a, b](x)

    # CHECK-NEXT: lit.call @decls::@"paramOverload2{{.*}}<:variadic<!Int> [a]>()
    paramOverload2[a]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload2{{.*}}<:variadic<!Int> [a, b]>()
    paramOverload2[a, b]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload2[decls::MyInt]()"
    paramOverload2[MyInt(a)]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload2[decls::MyInt,decls::MyInt]()"
    paramOverload2[MyInt(a), b]()

    # CHECK-NEXT: lit.call @decls::@"paramOverload2[{{.*}}<:variadic<!MyInt>
    paramOverload2[MyInt(a), b, c]()


struct VariadicStruct[*Ts: AnyTrivialRegType]:
    fn __init__(out self):
        pass

    @staticmethod
    fn param_func[i: Int]():
        pass


fn take_variadic_struct[*Ts: AnyTrivialRegType](a: VariadicStruct[*Ts]):
    pass


# CHECK-LABEL: lit.fn @"variadic_params()"
fn variadic_params():
    # CHECK-NEXT: call {{.*}}param_func[{{.*}}Int]()"<:variadic<type> [!Int, !FloatDyn], :!Int {4}>
    VariadicStruct[Int, FloatDyn].param_func[4]()
    # CHECK: call {{.*}}take_variadic_struct{{.*}}<:variadic<type> [!Int, !FloatDyn]
    take_variadic_struct(VariadicStruct[Int, FloatDyn]())


# Test that pointers don't get confused with by-ref arguments.
# CHECK-LABEL: lit.fn @"testPointerArgs{{.*}}(%ptr: !kgen.pointer<si32>) -> si32
fn testPointerArgs(ptr: __mlir_type.`!kgen.pointer<si32>`) -> __mlir_type.si32:
    # CHECK-NEXT: %0 = pop.load %ptr : !kgen.pointer<si32>
    return __mlir_op.`pop.load`[_type = __mlir_type.si32](ptr)


struct NoDebugInlineTest:
    # Two decorators stacked up
    @always_inline("nodebug")
    @staticmethod
    fn test():
        return


# CHECK-LABEL: lit.fn @"testAlwaysInlineNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
fn testAlwaysInlineNoDebug():
    pass


# CHECK-LABEL: lit.fn @"testNoInline
# CHECK-SAME: no_inline
@no_inline
fn testNoInline():
    pass


# CHECK-LABEL: lit.fn @"math{{.*}} always_inline_builtin
@always_inline("builtin")
fn math(a: __mlir_type.index, b: __mlir_type.index) -> __mlir_type.index:
    return __mlir_op.`index.add`(a, b)


# CHECK-LABEL: lit.fn @"useIt
fn useIt(a: __mlir_type.index) -> __mlir_type.index:
    # CHECK: %index3 = kgen.param.constant = <3>
    # CHECK: %0 = lit.call @decls::@"math(
    # CHECK: lit.return %0 : index
    return math(
        a,
        math(
            __mlir_op.`index.constant`[value = __mlir_attr.`1:index`](),
            __mlir_op.`index.constant`[value = __mlir_attr.`2:index`](),
        ),
    )


@always_inline("nodebug")
fn returnParameter[a: __mlir_type.index]() -> __mlir_type.index:
    return a


# CHECK-LABEL: lit.fn @"callReturnParam
fn callReturnParam() -> __mlir_type.index:
    # CHECK-NEXT: %0 = lit.call @decls::@"returnParameter[__mlir_type.index]()"<3>()
    # CHECK-NEXT: return %0
    return returnParameter[Int(3).value]()


# CHECK: lit.fn @"pleaseInline()"() -> index always_inline
@always_inline
fn pleaseInline() -> __mlir_type.index:
    return Int(1).value


# https://github.com/modularml/modular/issues/8500
struct AlwaysInlineByRef:
    @always_inline("nodebug")
    fn doByRef(mut self):
        pass


fn testInlineByRef(mut a: AlwaysInlineByRef):
    a.doByRef()


fn paramRefFunc[T: AnyTrivialRegType](x: T):
    pass


# CHECK-LABEL: lit.fn @"orvalueInferType()"
fn orvalueInferType():
    fn func(x: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: call {{.*}}paramRefFunc{{.*}}<:type !lit.generator<("x": index) -> index>>
    paramRefFunc(func)


# CHECK-LABEL: lit.fn @"kernel{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = ["nvvm.maxntid", {{.*}}#pop.array<x> : !pop.array<


@__llvm_metadata(
    `nvvm.maxntid`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel[x: Int]():
    pass


# CHECK-LABEL: lit.fn @"kernel1{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<
alias mname = "nvvm.maxntid"


@__llvm_metadata(
    mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel1[x: Int]():
    pass


# CHECK-LABEL: lit.fn @"kernel2{{.*}}"<x:
# CHECK-SAME: LLVMMetadataArray = [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, {{.*}}#pop.array<x> : !pop.array<


@__llvm_metadata(
    `mname`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel2[x: Int]():
    pass


# CHECK-LABEL: lit.fn @"llvm_arg_meta
# CHECK-SAME{LITERAL}: LLVMArgMetadataArray = [[], ["nvvm.grid_constant", unit, "myMeta", unit], [], [#kgen.unknown : !lit.struct<#StringLiteral <:string "nvvm.maxntid">>, #pop.array<x> : !pop.array<1, !Int>], []]
@__llvm_arg_metadata(b, `nvvm.grid_constant`, `myMeta`)
@__llvm_arg_metadata(c)
@__llvm_arg_metadata(
    d, mname=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn llvm_arg_meta[x: Int](a: Int, b: Int, c: Int, d: Int, e: Int):
    pass


fn alias_parametric_fn() -> StaticString:
    @parameter
    if True:
        return "nvvm.maxntid"
    else:
        return "rocdl.flat_work_group_size"


alias mname1 = _get_kgen_string[alias_parametric_fn()]()

# CHECK-LABEL: lit.fn @"kernel3{{.*}}"<x:


# TODO: Figure out how to get the value of the alias.
# CHECK-SAME: LLVMMetadataArray = [
# CHECK-SAME: #kgen.param.expr<data_to_str, {{.*}}alias_parametric_fn
# CHECK-SAME: #kgen.unknown : !lit.struct<#IntLiteral <:!pop.int_literal 128>>
@__llvm_metadata(
    mname1=128
)
fn kernel3[x: Int]():
    pass


# https://github.com/modular/mojo/issues/1152
# Allow mutable self argument when overloading operators using dunder methods
struct MutatingAdd:
    fn __add__(mut self, x: MutatingAdd):
        pass


# CHECK-LABEL: lit.fn @"testMutatingAdd
fn testMutatingAdd(owned a: MutatingAdd, b: MutatingAdd):
    # CHECK-NEXT: lit.call {{.*}}__add__{{.*}}(%a, %b)
    a + b


# CHECK-LABEL: lit.fn @"testContextSensitiveKeyword
# CHECK-SAME: (%out: !Int) -> !Int
fn testContextSensitiveKeyword(out x: Int, out: Int):
    # Check that we handle the result slot correctly.

    # CHECK-NEXT: %x = lit.var.decl "x"
    # CHECK-NEXT: lit.ref.store %out, %x
    # CHECK-NEXT: %0 = lit.load.consume %x
    # CHECK-NEXT: lit.return %0

    # out is an argument specifier, but that's a context sensitive keyword.
    # The identifier can be used like normal as well.
    x = out


##===----------------------------------------------------------------------===##
# Conventions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"ownedConventionMem
# CHECK-SAME: (%a: !lit.ref<!StructWithInit, mut {{.*}}> owned_in_mem,
# CHECK-SAME:  %b: !lit.ref<!StructWithInit, imm {{.*}}> read_mem)
fn ownedConventionMem(owned a: StructWithInit, b: StructWithInit):
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK: %1 = lit.ref.load [[AX]]
    _ = a.x
    # CHECK: [[BY:%.*]] = lit.ref.struct.ger %b[y]
    # CHECK: = lit.ref.load [[BY]]
    _ = b.y

    # It is ok to mutate owned values.
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}}4
    # CHECK-NEXT: lit.ref.store [[FOUR]], [[AX]]
    a.x = 4


@register_passable
struct RPStructWithInit:
    var x: Int
    var y: Int


@register_passable("trivial")
struct RPStructWithInitTrivial:
    var x: __mlir_type.index


# CHECK-LABEL: lit.fn @"ownedConventionReg
# CHECK-SAME: (%a: !lit.ref<!RPStructWithInit, mut *"a`"> owned_in_mem,
# CHECK-SAME:  %b: !lit.ref<!RPStructWithInit, imm *"b`1"> read_mem,
# CHECK-SAME:  %triv: !RPStructWithInitTrivial)
fn ownedConventionReg(
    owned a: RPStructWithInit,
    b: RPStructWithInit,
    triv: RPStructWithInitTrivial,
):
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK:  = lit.ref.load [[AX]]
    _ = a.x
    # CHECK: [[BY:%.*]] = lit.ref.struct.ger %b[y]
    # CHECK:  = lit.ref.load [[BY]]
    _ = b.y

    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK: [[ONE:%.*]]  = kgen.param.constant: !Int = <{1}>
    # CHECK: lit.ref.store [[ONE]], [[AX]]
    a.x = 1


struct BorrowStruct:
    fn testMethod(self):
        pass

    fn borrowedVarArgs(self, *x: BorrowStruct):
        pass


# CHECK-LABEL: callerFn
# CHECK-SAME: (%arg0: !lit.ref<{{.*}}> read_mem)
fn callerFn(arg0: BorrowStruct):
    # CHECK-NEXT: lit.call {{.*}}testMethod{{.*}}(%arg0)
    arg0.testMethod()

    # CHECK: %1 = pop.variadic.splat 2, %arg0
    # CHECK: lit.call {{.*}}borrowedVarArgs{{.*}}(%arg0,
    arg0.borrowedVarArgs(arg0, arg0)


##===----------------------------------------------------------------------===##
# Named Results
##===----------------------------------------------------------------------===##


struct SomeResultType:
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"named_result
# CHECK-SAME: %out: !lit.ref<!SomeResultType, {{.*}}> byref_result
# CHECK-SAME: namedResult = "out"
fn named_result(out out: SomeResultType):
    # CHECK-NEXT: call {{.*}}SomeResultType::@"__init__{{.*}}(%out)
    out = SomeResultType()
    # CHECK: lit.return %none
    return
    # CHECK-NEXT: lit.end_fn


# CHECK-LABEL: lit.fn @"named_result_return_expr
fn named_result_return_expr(out out: SomeResultType):
    # CHECK-NEXT: call {{.*}}SomeResultType::@"__init__{{.*}}(%out)
    return SomeResultType()


##===----------------------------------------------------------------------===##
# Default arguments and variadics.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"defaultArgument
# CHECK-SAME: %c: !Int = {5})
fn defaultArgument(a: Int, b: Int = 3, c: Int = 5) -> Int:
    return a + b


# CHECK-LABEL: lit.fn @"callDefaultArgument
fn callDefaultArgument(x: Int) -> Int:
    # CHECK: [[ARG1:%.*]] = kgen.param.constant{{.*}}3
    # CHECK-NEXT: [[ARG2:%.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: lit.call {{.*}}defaultArgument{{.*}}(%x, [[ARG1]], [[ARG2]])
    # CHECK-NEXT: lit.ref.store {{.*}}, %a
    var a = defaultArgument(x)

    # CHECK-NEXT: %b = lit.var.decl
    # CHECK-NEXT: %[[ARG2:.*]] = kgen.param.constant{{.*}}5
    # CHECK-NEXT: lit.call {{.*}}defaultArgument{{.*}}(%x, %x, %[[ARG2]])
    var b = defaultArgument(x, x)
    return a + b


# CHECK-LABEL: lit.fn @"defaultArgumentReferencesParameter
# CHECK-SAME: (%a: !Int = {value = add(#lit.struct.extract<:!Int p, "value">, 87)})
fn defaultArgumentReferencesParameter[p: Int](a: Int = p + 87) -> Int:
    return a


struct MemoryType:
    var value: Int

    @implicit
    fn __init__(out self, value: Int):
        self.value = value

    # MOCO-1445: throwing implicit conversions.
    @implicit
    fn __init__(out self, value: String) raises:
        self.value = 4

    # Default arguments and variadics.
    @implicit
    fn __init__(
        out self, value: SomeResultType, stuff: Int = 4, *other: String
    ):
        self.value = 4


# CHECK-LABEL: lit.fn @"defaultArgumentNonRegisterType
# CHECK-SAME: read_mem = apply_result_slot({{.*}}__init__
fn defaultArgumentNonRegisterType(a: MemoryType = 1):
    pass


# CHECK-LABEL: lit.fn @"callNonRegisterDefaultArg
fn callNonRegisterDefaultArg():
    # CHECK: %[[ANON:.*]] = lit.var.decl "anonymous*" synth : !lit.ref<!MemoryType, mut *"anonymous*`">
    # CHECK: %[[VALUE:.*]] = kgen.param.materialize: !MemoryType = <apply_result_slot({{.*}}1}
    # CHECK: lit.ref.store %[[VALUE]], %[[ANON]]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A
    # CHECK: call {{.*}}defaultArgumentNonRegisterType{{.*}}([[IMMREF]])
    defaultArgumentNonRegisterType()
    # CHECK: lit.alias.decl *"none{{.*}}": none = <apply({{.*}}defaultArgumentNonRegisterType
    # CHECK-SAME: store_to_mem(apply_result_slot({{.*}}MemoryType::@"__init__{{.*}}1}
    alias none = defaultArgumentNonRegisterType()


# CHECK: lit.fn @"referencesDefaultArgumentFunction
fn referencesDefaultArgumentFunction():
    # CHECK: %f = lit.var.decl "f"
    # CHECK: lit.ref.store %0, %f
    var f = defaultArgument


# CHECK-LABEL: lit.struct.decl @Outer<X:
struct Outer[X: Int]:
    # CHECK: lit.fn @"nested
    # CHECK-SAME: %x: !Int = X)
    fn nested(self, x: Int = X):
        pass


# CHECK-LABEL: lit.fn @"variadics({{.*}}Int*)"(%a: !kgen.variadic<!Int> var)
fn variadics(*a: Int):
    # CHECK: lit.call {{.*}}VariadicList{{.*}}__init__
    pass


fn parameterizedVariadic[T: __mlir_type.`!kgen.type`](*args: T):
    pass


struct ParameterizedStruct[T: __mlir_type.`!kgen.type`]:
    @implicit
    fn __init__(out self, *args: T):
        pass


struct VarArgsParameterizedStruct[*Is: Int]:
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"callVariadic{{.*}}"<p: !Int>
fn callVariadic[p: Int](x: Int):
    # CHECK: %variadic = kgen.param.constant: variadic<!Int> = <[]>
    # CHECK: call @decls::@"variadics({{.*}}Int*)"(%variadic)
    variadics()
    # CHECK: %[[C7:.*]] = kgen.param.constant{{.*}}7
    # CHECK: %[[C11:.*]] = kgen.param.constant{{.*}}11
    # CHECK: %[[VARIADIC:.*]] = pop.variadic.create [%[[C7]], %[[C11]]]
    # CHECK: call @decls::@"variadics({{.*}}Int*)"(%[[VARIADIC]])
    variadics(7, 11)
    # CHECK: %[[VAR:.*]] = pop.variadic.splat 1, %x
    # CHECK: call @decls::@"variadics({{.*}}Int*)"(%[[VAR]])
    variadics(x)
    # CHECK: %[[CST:.*]] = kgen.param.constant: !Int
    # CHECK: %[[VAR:.*]] = pop.variadic.create [%x, %[[CST]]]
    # CHECK: call @decls::@"variadics({{.*}}Int*)"(%[[VAR]])
    variadics(x, 1)

    # CHECK: @"variadics({{.*}}Int*)", []
    alias EmptyVariadic = variadics()
    # CHECK: @"variadics({{.*}}Int*)", [p, {1}]
    alias NonEmptyVariadic = variadics(p, 1)

    # CHECK: @"parameterizedVariadic{{.*}}"<:type !Int>
    parameterizedVariadic(1, 2)
    # CHECK: lit.call {{.*}}@ParameterizedStruct::@"__init__({{.*}}<:type !Int>
    _ = ParameterizedStruct(3)
    # CHECK: lit.call {{.*}}@VarArgsParameterizedStruct::@"__init__({{.*}}<:variadic<!Int> [{4}, {5}]>
    _ = VarArgsParameterizedStruct[4, 5]()
    # CHECK: lit.call {{.*}}@VarArgsParameterizedStruct::@"__init__({{.*}}<:variadic<!Int> []>
    _ = VarArgsParameterizedStruct()


# COM: Test variadic arguments in a parameter context.
@value
struct MemStruct:
    alias t = 5


fn variadic_mem_only(*values: MemStruct) -> Int:
    return 0


# CHECK-LABEL: lit.fn @"test_variadic_mem_only{{.*}}"<x: !MemStruct, y: !MemStruct>
fn test_variadic_mem_only[x: MemStruct, y: MemStruct]():
    # CHECK: lit.alias.decl {{.*}}: !Int = <apply(
    # CHECK-SAME: :!lit.generator<[1]("values": !kgen.variadic<!lit.ref<!MemStruct, imm {}>, read_mem> var) -> !Int> {{.*}}::@"variadic_mem_only({{.*}}::MemStruct*)"
    # CHECK-SAME: [store_to_mem(x), store_to_mem(y)]
    alias b = variadic_mem_only(x, y)


##===----------------------------------------------------------------------===##
# raises specifier.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"defAlwaysRaises()"[{{.*}}](?, %__error__: {{.*}}, %__result__: {{.*}}) throws -> i1 attributes {isDef
def defAlwaysRaises() -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}{0}
    # CHECK: lit.ref.store [[RESULT]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return 0


# CHECK-LABEL: lit.fn @"fnThatRaises()"{{.*}} throws -> i1
fn fnThatRaises() raises -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}{0}
    # CHECK-NEXT: lit.ref.store [[RESULT]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return 0


# CHECK-LABEL: lit.fn @"raisesReturnsNone()"{{.*}} throws -> i1
fn raisesReturnsNone() raises:
    # CHECK-NEXT: %none = kgen.param.constant: none
    # CHECK-NEXT: lit.ref.store %none, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    # CHECK-NEXT: lit.end_fn
    pass


# COM: We can return an variant of error and index in a non-throwing function.
# CHECK-LABEL: lit.fn @"raisesReturnsVariant()"() -> !kgen.variant<!Error, index>
fn raisesReturnsVariant() -> __mlir_type[`!kgen.variant<`, Error, `, index>`]:
    return __mlir_op.`kgen.variant.create`[
        _type = __mlir_type[`!kgen.variant<`, Error, `, index>`],
        index = Int(1).value,
    ](Int(1).value)


# CHECK-LABEL: lit.fn @"raise_and_return{{.*}} throws -> i1
fn raise_and_return(a: Error) raises -> Error:
    # COM: True result indicates an error.
    # CHECK: [[ERR:%.*]] = lit.call {{.*}}Error::@"__init__
    # CHECK-NEXT: lit.ref.store [[ERR]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return Error()


@value
@register_passable("trivial")
struct RaisingGetterSetter:
    fn __getitem__(self, i: Int) raises -> FloatDyn:
        return 1.0

    fn __setitem__(mut self, i: Int, v: FloatDyn) raises:
        pass


fn test_raising_computed_getter() raises:
    var a = RaisingGetterSetter()[2]


##===----------------------------------------------------------------------===##
# Structs
##===----------------------------------------------------------------------===##


fn forward_ref(x: EmptyStruct):
    pass


# CHECK-LABEL: lit.struct.decl @EmptyStruct({{.*}}) register_passable
@register_passable
struct EmptyStruct:
    pass


# CHECK-LABEL: lit.struct.decl @OneLineStruct<size: !Int>
struct OneLineStruct[size: Int]:
    pass
    pass


# CHECK-LABEL: lit.struct.decl @StructWithInit
struct StructWithInit:
    var x: Int
    var y: Int

    # CHECK: lit.fn @"__init__({{.*}}Int)"
    # CHECK-SAME: %self: !lit.ref<!StructWithInit, mut {{.*}}> byref_result)
    @implicit
    fn __init__(out self, a: Int):
        # CHECK: %0 = lit.ref.struct.ger %self[x]
        # CHECK: lit.ref.store %a, %0
        self.x = a
        # CHECK: [[XP:%.*]] = lit.ref.struct.ger %self[x]
        # CHECK: [[YP:%.*]] = lit.ref.struct.ger %self[y]
        # CHECK: [[XT:%.*]] = lit.ref.load [[XP]]
        # CHECK: lit.ref.store [[XT]], [[YP]]
        self.y = self.x
        # CHECK-NEXT: kgen.param.constant: none
        # CHECK-NEXT: lit.return
        return

    # Not very useful, but this form also works, so test it.
    # CHECK: lit.fn @"__init__
    # CHECK-SAME: %self: !lit.ref<!StructWithInit, mut {{.*}}> byref_result)
    fn __init__(out self, a: Int, b: Int):
        # CHECK: hlcf.elif
        if a == b:
            # CHECK:  lit.call {{.*}}__init__{{.*}}(%a, %self)
            self = StructWithInit(a)
        else:
            # CHECK: [[XP:%.*]] = lit.ref.struct.ger %self[x]
            # CHECK: lit.ref.store %a, [[XP]]
            # CHECK: [[YP:%.*]] = lit.ref.struct.ger %self[y]
            # CHECK: lit.ref.store %b, [[YP]]
            self.x = a
            self.y = b

    fn __init__(out self):
        self = Self(0)


# CHECK-LABEL: lit.struct.decl @StructExample
@register_passable
struct StructExample:
    fn __copyinit__(out self, other: Self):
        pass

    fn __init__(out self):
        pass

    # CHECK: lit.fn @"maybe_static({{.*}}Int)"(%x: !Int) {{.*}}isStatic
    @staticmethod
    fn maybe_static(x: Int):
        # CHECK: %0 = {{.*}}{4}
        # CHECK: lit.call @decls::@StructExample::@"maybe_static{{.*}}"(%0)
        StructExample.maybe_static(4)
        pass

    # This isn't static.
    # CHECK: lit.fn @"maybe_static
    fn maybe_static(self, x: EmptyStruct):
        # CHECK: %0 = {{.*}}{4}
        # CHECK: lit.call @decls::@StructExample::@"maybe_static{{.*}}"(%0)
        StructExample.maybe_static(4)
        pass

    # CHECK: lit.fn @"mutatingMethod{{.*}}(%self: !lit.ref<!StructExample, mut {{.*}}> mut) -> !kgen.none
    fn mutatingMethod(mut self):
        pass


# CHECK-LABEL: lit.fn @"callMaybeStatic
fn callMaybeStatic(a: Int, b: EmptyStruct):
    # CHECK-NEXT: lit.call @decls::@StructExample::@"maybe_static{{.*}}(%a)
    StructExample.maybe_static(a)

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@StructExample::@"__init__{{.*}}()
    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[TMP]], [[ANONSE]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[ANONSE]]
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}([[TMP]], %b)
    StructExample.maybe_static(StructExample(), b)

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@"__init__{{.*}}()
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}(%a)
    StructExample().maybe_static(a)

    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}@StructExample::@"__init__{{.*}}()
    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.ref.store [[TMP]], [[ANONSE]]
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut [[ANONSE]]
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}([[TMP]], %b)
    StructExample().maybe_static(b)


# CHECK-LABEL: lit.fn @"initializersAsFunctions
# See that we can take the address of initializers without a thunk.
fn initializersAsFunctions():
    # Register passable trivial.
    # CHECK-NEXT: %fn_ptr1 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure[{{.*}}:!lit.generator<("_a": !Int) -> !MyInt> @decls::@MyInt::@"__init__(::Int)")]()
    # CHECK-NEXT: lit.ref.store [[TMP]], %fn_ptr1
    var fn_ptr1: fn (Int) -> MyInt = MyInt.__init__

    # Register passable non-trivial.

    # CHECK-NEXT: %fn_ptr2 = lit.var.decl "fn_ptr2"
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure[!lit.generator<() -> !StructExample>: @decls::@StructExample::@"__init__()"]()
    # CHECK-NEXT: lit.ref.store [[TMP]], %fn_ptr2
    var fn_ptr2: fn () -> StructExample = StructExample.__init__

    # CHECK-NEXT: %fn_ptr4 = lit.var.decl "fn_ptr4"
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure{{.*}}@StructExample::@"__copyinit__(decls::StructExample)")]()
    # CHECK-NEXT: lit.ref.store [[TMP]], %fn_ptr4
    var fn_ptr4: fn (
        StructExample
    ) -> StructExample = StructExample.__copyinit__

    # Memory
    # CHECK-NEXT: %fn_ptr5 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure[{{.*}}:!lit.generator<[1]("a": !Int, ?, "self": !lit.ref<!StructWithInit, mut *[0,0]> byref_result) -> !kgen.none> @decls::@StructWithInit::@"__init__(::Int)")
    # CHECK-NEXT: lit.ref.store [[TMP]], %fn_ptr5
    var fn_ptr5: fn (Int) -> StructWithInit = StructWithInit.__init__


# CHECK-LABEL: lit.struct.decl @DelegatingInitMem
# Issue #12042
struct DelegatingInitMem:
    var value: Int

    # CHECK: lit.fn @"__init__{{.*}}({{.*}}%self
    @implicit
    fn __init__(out self, value: Bool):
        # CHECK: lit.call @{{.*}}__init__{{.*}}(%0, %self)
        self = Self(42)

    @implicit
    fn __init__(out self, value: Int):
        self.value = value


# External issue #260
fn nameOutsideStruct(x: Int, y: Int):
    pass


struct ShadowsOuterName:
    fn nameOutsideStruct(self):
        nameOutsideStruct(1, 2)


struct LegacyInOutInit:
    # This should be accepted for compatibility, but "out" is the preferred
    # spelling.
    fn __init__(out self):
        pass


##===----------------------------------------------------------------------===##
# Struct @value decorator
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.struct.decl @ValueMem(!AnyType_Copyable_ExplicitlyCopyable_Movable_UnknownDestructibility)
# CHECK: move :!lit.generator<[2]({{.*}} owned_in_mem, |, ?, {{.*}} byref_result) {{.*}}ValueMem::@"__moveinit__
@value
struct ValueMem:
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
# CHECK-SAME: ) -> !kgen.none always_inline_no_debug attributes {isStatic, isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
# CHECK-NEXT: %[[PA:.*]] = lit.ref.struct.ger %self[a]
# CHECK-NEXT: lit.ref.store %a, %[[PA]]
# CHECK-NEXT: %[[PB:.*]] = lit.ref.struct.ger %self[b]
# CHECK-NEXT: [[TMP:%.*]] = lit.load.consume %b
# CHECK-NEXT: lit.ref.store [[TMP]], %[[PB]]


# CHECK-LABEL: lit.struct.decl @ValueMemHasCopy(!AnyType_Copyable_ExplicitlyCopyable_Movable_UnknownDestructibility)
@value
struct ValueMemHasCopy:
    var a: Int
    var b: StructExample

    fn __copyinit__(out self, other: Self):
        self.a = other.a
        self.b = other.b


# CHECK-LABEL: lit.struct.decl @ValueMemHasMove(!AnyType_Copyable_ExplicitlyCopyable_Movable_UnknownDestructibility)
@value
struct ValueMemHasMove:
    var a: Int
    var b: StructExample

    fn __moveinit__(out self, owned other: Self):
        self.a = other.a
        self.b = other.b


# CHECK-LABEL: lit.struct.decl @ValueRegTrivial
# CHECK-SAME: (!AnyType_Copyable_ExplicitlyCopyable_Movable_UnknownDestructibility) register_passable_trivial

# CHECK: lit.fn @"__copyinit__{{.*}}_thunk"[{{.*}}](%0[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> read_mem,
# CHECK-SAME: %1[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> byref_result) -> !kgen.none always_inline_no_debug
# CHECK-NEXT: [[V0:%.*]] = lit.ref.load %0 : <!ValueRegTrivial
# CHECK-NEXT: lit.ref.store [[V0]], %1
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: kgen.return %none : !kgen.none

# CHECK: lit.fn @"__moveinit__{{.*}}_thunk"[{{.*}}](%0[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> owned_in_mem,
# CHECK-SAME: %1[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> byref_result)
# CHECK-NEXT: [[V0:%.*]] = lit.load.consume %0
# CHECK-NEXT: lit.ref.store [[V0]], %1
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: kgen.return %none : !kgen.none


@value
@register_passable("trivial")
struct ValueRegTrivial:
    var a: __mlir_type.index


# CHECK-LABEL: lit.struct.decl @ValueReg
@value
@register_passable
struct ValueReg:
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
# CHECK-LABEL: lit.struct.decl @Foo(!AnyType_Copyable_ExplicitlyCopyable_Movable_UnknownDestructibility) attributes
@value
struct Foo:
    var a: Int
    var self: Int


# CHECK: lit.fn @"__init__{{.*}}(%a: !Int, %self: !Int, ?, %self_0[self]: !lit.ref<!Foo, mut {{.*}}> byref_result)


# CHECK-LABEL: lit.struct.decl @ParamVarArg<I: variadic<!Int> var>
@value
@register_passable("trivial")
struct ParamVarArg[*I: Int]:
    pass


# CHECK-LABEL: lit.struct.decl @TraitMember
@value
struct TraitMember[T: Copyable]:
    var value: T
    # CHECK: lit.fn @"__moveinit__
    # CHECK: call{{.*}}__copyinit__
    # CHECK: lit.fn @"__copyinit__
    # CHECK: call{{.*}}__copyinit__


# CHECK: lit.fn @"notSynthetic{{.*}}(%self: !lit.ref<!NotSynthetic, imm {{.*}}> read_mem) -> !kgen.none attributes {sourceName = "notSynthetic", specialFnKind = 0 : i8}
# CHECK: lit.fn @"__moveinit__{{.*}}isSynthetic
# CHECK: lit.fn @"__copyinit__{{.*}}isSynthetic
# CHECK: lit.fn @"__init__{{.*}}isSynthetic
@value
struct NotSynthetic:
    var member: __mlir_type.`index`

    fn notSynthetic(self):
        pass


# CHECK-LABEL: lit.struct.decl @VarArgInit
@value
@register_passable("trivial")
struct VarArgInit:
    var a: Int

    # CHECK: lit.fn @"__init__(decls::ValueMem*)"{{.*}}(%values: !kgen.variadic<!lit.ref<!ValueMem, imm {{.*}}>, read_mem> var
    # The argument is intentionally memory-only.
    @implicit
    fn __init__(out self, *values: ValueMem):
        self.a = 42

    # CHECK: lit.fn @"__init__(::Int)"(%a: !Int) -> !VarArgInit


# COM: Body resolution of `Node` will recurse on itself. Make sure that the
# COM: trait requirements for Copyable and Movable are generated early.
struct BoxCopyable[T: Copyable]:
    pass


@value
struct Node:
    var id: RecursiveCopyable.ID


# CHECK-LABEL: lit.struct.decl @RecursiveCopyable
struct RecursiveCopyable:
    alias ID = Int
    # CHECK: lit.struct.field recurse
    # CHECK-SAME: "__copyinit__"
    var recurse: BoxCopyable[Node]


# CHECK-LABEL: lit.struct.decl @RaisingMemberwiseInit
@value
struct RaisingMemberwiseInit:
    var x: Int

    # CHECK-LABEL: lit.fn @"__init__{{.*}} throws
    fn __init__(out self, x: Int) raises:
        pass


##===----------------------------------------------------------------------===##
# async/await
##===----------------------------------------------------------------------===##


@register_passable("trivial")
struct Container[T: AnyType]:
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        T,
        `>`,
    ]
    var address: Self._mlir_type

    fn __init__(out self):
        self.address = __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]


async fn load(server_ptr: Container[__mlir_type.index]):
    pass


# CHECK-LABEL: lit.fn @"awaitSomething()"
async fn awaitSomething():
    var ptr = Container[__mlir_type.index]()
    # CHECK: [[CORO:%.*]] = lit.call {{.*}}@Coroutine::@"__init__{{.*}}<:!AnyType [{{.*}}], :origin.set {}>(%{{.*}}) :
    # CHECK-SAME: !lit.generator<("handle": !co.routine) -> !lit.struct<#Coroutine <:!AnyType
    await load(ptr)


# CHECK-LABEL: lit.fn @"coroutine
# CHECK-SAME: [mut [[LT:.*]]](?, %__result__: !lit.ref<!Int, mut [[LT]]> byref_result) async -> !kgen.none
async fn coroutine() -> Int:
    # CHECK: lit.ref.store %0, %__result__
    # CHECK: lit.return %none
    return 0


# CHECK-LABEL: lit.struct.decl @StructWithAsync
struct StructWithAsync:
    # CHECK-LABEL: lit.fn @"do_something{{.*}}({{.*}}) async
    async fn do_something(self: StructWithAsync):
        # CHECK-NEXT: %[[CORO:.*]] = lit.async.call[!lit.generator<[1](?, "__result__": !lit.ref<!Int, mut *[0,0]> byref_result) async -> !kgen.none>: @decls::@"coroutine()"][imm {}]()
        # CHECK: lit.call {{.*}}@Coroutine::@"__init__{{.*}}<:!AnyType [!Int, {{.*}}], :origin.set {}>(%[[CORO]])
        _ = coroutine()


# CHECK-LABEL: lit.fn @"call_struct_async
# CHECK-SAME: [imm [[LT:.*]], mut {{.*}}]{{.*}}) async -> !kgen.none
async fn call_struct_async(f: StructWithAsync):
    # CHECK-NEXT: lit.async.call[!lit.generator<[2]({{.*}}, "__result__":{{.*}}) async -> !kgen.none>: @{{.*}}][imm [[LT]], imm {}](%f)
    _ = f.do_something()


struct Awaitable:
    fn __init__(out self):
        pass

    fn __await__(mut self) -> Int:
        return 0


# CHECK-LABEL: lit.fn @"awaitable()"
fn awaitable() -> Int:
    # CHECK: call {{.*}}@Awaitable::@"__await__{{.*}}(%aw)
    var aw = Awaitable()
    return await aw


# COM: https://github.com/modular/mojo/issues/951
@always_inline
async fn inline_async() -> Int:
    return 0


# CHECK-LABEL: lit.fn @"use_inline_async()"
async fn use_inline_async() -> Int:
    # CHECK: [[ASYNC_RESULT:%.*]] = lit.async.call{{.*}}inline_async
    # CHECK: [[TMP:%.*]] = lit.call {{.*}}Coroutine{{.*}}__init__{{.*}}([[ASYNC_RESULT]]) :
    # CHECK: lit.ref.store [[TMP]], [[CORO:%.*]] : <
    # CHECK: lit.call {{.*}}Coroutine{{.*}}__await__{{.*}}([[CORO]], %__result__)
    return await inline_async()


async fn capture_byref(mut x: Awaitable, y: Awaitable):
    pass


@value
@register_passable
struct LifetimeAccess[origin: __mlir_type.`!lit.origin<1>`]:
    pass


async fn lifetime_access(owned x: LifetimeAccess[_]):
    pass


# CHECK-LABEL: lit.fn @"coroutine_origins
fn coroutine_origins():
    # CHECK: var.decl "x" var : {{.*}}mut [[X_LT:.*]]>
    var x: Awaitable
    # CHECK: var.decl "y" var : {{.*}}mut [[Y_LT:.*]]>
    var y: Awaitable
    # CHECK: [[Y_IMM:%.*]] = lit.ref.immut %y
    # CHECK: [[CORO:%.*]] = lit.async.call[!lit.generator<[3]("x": !lit.ref<!Awaitable, mut *[0,0]> mut, "y": !lit.ref<!Awaitable, imm *[0,1]> read_mem, ?, "__result__": !lit.ref<none, mut *[0,2]> byref_result) async -> !kgen.none>
    # CHECK-SAME: [mut [[X_LT]], muttoimm [[Y_LT]], imm {}](%x, [[Y_IMM]])
    # CHECK: lit.call {{.*}}Coroutine::@"__init__{{.*}}<:!AnyType [none, {{.*}}], :origin.set {mut [[X_LT]], mut [[Y_LT]]}>([[CORO]])
    var coro = capture_byref(x, y)

    # CHECK: lit.async.call[!lit.generator<[2]("x": !lit.ref<@decls::@LifetimeAccess<:origin<1> [[Y_LT]]>,
    # CHECK-SAME: mut *[0,0]{{.*}}) async -> !kgen.none>: {{.*}}lifetime_access{{.*}}<:origin<1> [[Y_LT]]>]
    # CHECK: Coroutine<:!AnyType [none, {{.*}}], :origin.set {{{.*}}, mut [[Y_LT]]}>
    var access = lifetime_access(LifetimeAccess[__origin_of(y)]())


# CHECK-LABEL: lit.fn @"mem_result{{.*}}(?, %__result__: !lit.ref<!Awaitable, {{.*}}> byref_result) async -> !kgen.none
async fn mem_result() -> Awaitable:
    # CHECK: [[CORO:%.*]] = lit.async.call[{{.*}}mem_result()"][imm {}]()
    # CHECK: lit.call {{.*}}@Coroutine::@"__init__{{.*}}([[CORO]])
    var coro = mem_result()


# CHECK-LABEL: lit.fn @"mem_raises{{.*}}(?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<!Int, {{.*}}> byref_result) throws|async -> i1
async fn mem_raises() raises -> Int:
    # CHECK: [[CORO:%.*]] = lit.async.call[{{.*}}mem_raises()"][imm {}, imm {}]()
    # CHECK: lit.call {{.*}}@RaisingCoroutine::@"__init__{{.*}}([[CORO]])
    var coro = mem_raises()


# CHECK-LABEL: lit.fn @"async_closure_capture
fn async_closure_capture(x: String):
    @parameter
    # CHECK: lit.fn *"capture_it
    async fn capture_it():
        _ = x

    # CHECK: Coroutine<{{.*}}{imm *"x`
    # CHECK-NEXT: lit.async.call[{{.*}}capture_it
    var coro = capture_it()


##===----------------------------------------------------------------------===##
# Nested Functions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"topLevelFunction()"
fn topLevelFunction() -> Int:
    var a = 0

    # CHECK: lit.fn *"nestedFunction()"
    @parameter
    fn nestedFunction() -> Int:
        # CHECK-NEXT: lit.ref.load %a
        return a

    # CHECK: lit.alias.decl *"b{{.*}}": !lit.generator<:{mut *"a`"}:() capturing -> !Int> = <*"nestedFunction()">
    alias b = nestedFunction
    # CHECK: call[!lit.generator<:{mut *"a`"}:() capturing -> !Int>: *"nestedFunction()"]()
    return nestedFunction()


# CHECK-LABEL: lit.struct.decl @SomeStruct
struct SomeStruct:
    # CHECK-LABEL: @"someMethod({{.*}})"
    fn someMethod(self) -> Int:
        var a = 0

        # CHECK: lit.fn *"nestedFunction()"
        @parameter
        fn nestedFunction() -> Int:
            # CHECK-NEXT: lit.ref.load %a
            return a

        # CHECK: lit.alias.decl *"b{{.*}}": !lit.generator<:{mut [[A_LT:\*"a`.*"]]}:() capturing -> !Int> = <*"nestedFunction()">
        alias b = nestedFunction
        # CHECK: call[!lit.generator<:{mut [[A_LT]]}:() capturing -> !Int>: *"nestedFunction()"]()
        return nestedFunction()


# CHECK-LABEL: lit.fn @"closureParameter[fn() capturing -> __mlir_type.index]()"
# CHECK-SAME: capturing ->
fn closureParameter[func: fn () capturing -> __mlir_type.index]():
    pass


# CHECK-LABEL: lit.fn @"closureParameterCaptures
# CHECK-SAME: :*(0,0):
# CHECK-SAME: func: !lit.generator<:origins:() capturing -> !kgen.none>
fn closureParameterCaptures[
    origins: OriginSet, //, func: fn () capturing [origins] -> None
]():
    pass


@register_passable("trivial")
struct HasParam[p: Index]:
    pass


fn closureParameterInference[
    p: Index, //, f: fn () capturing -> None
](arg: HasParam[p]):
    pass


@register_passable("trivial")
struct HasLifetimeParam[p: MutableOrigin]:
    pass


# CHECK-LABEL: lit.fn @"explicitLifetime
# CHECK-SAME: :{mut *(0,0)}:
fn explicitLifetime[lt: MutableOrigin, //, arg: HasLifetimeParam[lt]]():
    pass


# CHECK-LABEL: lit.fn @"inaccessibleImplicitLifetimeParam
# CHECK-SAME: "<?, *"p`": origin<1>>(%arg:
fn inaccessibleImplicitLifetimeParam(arg: HasLifetimeParam):
    pass


# CHECK-LABEL: lit.struct.decl @CapturingStruct
struct CapturingStruct[a: Index]:
    @staticmethod
    fn takeClosure[
        origins: OriginSet, //,
        f: fn () capturing [origins] -> None,
    ]():
        pass


# CHECK-LABEL: lit.trait.decl @CapturingTrait
trait CapturingTrait:
    # CHECK: lit.fn @"takeClosure{{.*}}:*(0,0):
    fn takeClosure[
        origins: OriginSet, //,
        f: fn () capturing [origins] -> None,
    ](self):
        ...


# CHECK-LABEL: lit.struct.decl @CapturingStructTrait
@register_passable
struct CapturingStructTrait(CapturingTrait):
    # CHECK: lit.fn @"takeClosure{{.*}}:*(0,0):
    fn takeClosure[
        origins: OriginSet, //,
        f: fn () capturing [origins] -> None,
    ](self):
        pass


# CHECK-LABEL: lit.fn @"inferCaptureOrigins
fn inferCaptureOrigins[
    lt: MutableOrigin, param: HasLifetimeParam[lt]
](mut x: Index, mut y: Index, arg: HasParam):
    @parameter
    fn bareFunc():
        pass

    @parameter
    fn captureSomething():
        _ = x

    # CHECK: call {{.*}}closureParameterCaptures{{.*}}<:origin.set {},
    # CHECK-SAME: !lit.generator<() capturing -> !kgen.none>
    closureParameterCaptures[bareFunc]()
    # CHECK: call {{.*}}closureParameterCaptures{{.*}}<:origin.set {mut *"x`"},
    # CHECK-SAME: !lit.generator<:{mut *"x`"}:() capturing -> !kgen.none>
    closureParameterCaptures[captureSomething]()
    # CHECK: call {{.*}}closureParameterInference{{.*}}<*"p`{{.*}}",
    # CHECK-SAME: rebind(:!lit.generator<:{mut *"x`"}:{{.*}} *"captureSomething
    closureParameterInference[captureSomething](arg)

    # CHECK: lit.alias.decl *"unboundSet{{.*}} !lit.generator<<{{.*}}>:*(0,0):
    alias unboundSet = closureParameterCaptures
    # CHECK: lit.alias.decl *"boundSet{{.*}} !lit.generator<:{mut *"x`"}
    alias boundSet = closureParameterCaptures[captureSomething]

    # CHECK: lit.alias.decl *"unboundSingleParam{{.*}} !lit.generator<<{{.*}}>:{mut *(0,0)}
    alias unboundSingleParam = explicitLifetime
    # CHECK: lit.alias.decl *"boundSingleParam{{.*}} !lit.generator<:{mut lt}
    alias boundSingleParam = explicitLifetime[param]

    # CHECK: lit.alias.decl *"memberFunction{{.*}} !lit.generator<<{{.*}}>:*(0,1):
    alias memberFunction = CapturingStruct.takeClosure

    # CHECK: lit.fn *"captureWithClosure
    # CHECK-SAME: :{mut |*(0,0)|, mut *"y`{{.*}}"}:
    @parameter
    fn captureWithClosure[
        lts: OriginSet, //, f: fn () capturing [lts] -> None
    ]():
        _ = y

    # CHECK: lit.alias.decl *"boundClosure{{.*}} !lit.generator<:{mut *"x`", mut *"y`{{.*}}"}
    alias boundClosure = captureWithClosure[captureSomething]


# CHECK-LABEL: lit.fn @"testParameterCapture
fn testParameterCapture(mut x: Index, mut y: Index):
    # CHECK: lit.fn *"capture()":{mut *"x`"}
    @parameter
    fn capture():
        _ = x

    # CHECK: lit.fn *"do_it()":{mut *"x`", mut *"y`
    @parameter
    fn do_it():
        _ = y
        capture()


# CHECK-LABEL: lit.fn @"topLevelParamFn[__mlir_type.index]()"<a_param>
fn topLevelParamFn[a_param: __mlir_type.index]():
    # CHECK: lit.fn *"nestedFunction[__mlir_type.index]()"<b_param>
    fn nestedFunction[b_param: __mlir_type.index]():
        return

    # CHECK: lit.alias.decl *"thinref{{.*}}": !lit.generator<<"b_param": index>() -> !kgen.none> = <*"nestedFunction[__mlir_type.index]()">
    alias thinref = nestedFunction
    # CHECK: call[{{.*}}: bind_params(:!lit.generator<<"b_param": index>() -> !kgen.none> *"nestedFunction[__mlir_type.index]()", 2)]()
    nestedFunction[Int(2).value]()

    var value = 0

    @__copy_capture(value)
    @parameter
    fn capturingNestedFunction() -> Int:
        return value

    # CHECK: lit.alias.decl *"fatRef{{.*}}": !lit.generator<() capturing -> !Int> = <*"capturingNestedFunction()">
    alias fatRef = capturingNestedFunction


struct SomeParamStruct[c_param: Int]:
    # CHECK-LABEL: lit.fn @"topLevelParamFn{{.*}}<a_param: !Int>
    fn topLevelParamFn[a_param: Int](self):
        # CHECK: lit.fn *"nestedFunction{{.*}}"<b_param: !Int>
        fn nestedFunction[b_param: Int]():
            return

        # CHECK: lit.alias.decl *"reff{{.*}}": !lit.generator<<"b_param": !Int>() -> !kgen.none> = <*"nestedFunction[{{.*}}Int]()">
        alias reff = nestedFunction
        # CHECK: call[{{.*}}: bind_params(:!lit.generator<<"b_param": !Int>() -> !kgen.none> *"nestedFunction[{{.*}}Int]()", {{.*}}2{{.*}})]()
        nestedFunction[2]()


##===----------------------------------------------------------------------===##
# Exported Functions
##===----------------------------------------------------------------------===##


@export("my_named_export", ABI="C")
# CHECK: lit.fn export C @"export_me()"
# CHECK-SAME: linkageName = "my_named_export"
def export_me() -> None:
    ...


@export
# CHECK: lit.fn export @"not_c_exported()"
fn not_c_exported():
    pass


struct Thing:
    # CHECK: lit.fn export @"member
    @export
    fn member(self):
        pass


##===----------------------------------------------------------------------===##
# Decorators
##===----------------------------------------------------------------------===##


fn elementwise():
    return


fn register(a: StringLiteral):
    return


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


# CHECK-LABEL: lit.alias.decl
# CHECK-SAME: deprecationWarning = "alias"
@deprecated("alias")
alias deprecated_alias = 1


##===----------------------------------------------------------------------===##
# Implicit origins for result slots.
##===----------------------------------------------------------------------===##


struct MyStruct:
    fn __init__(out self):
        pass


# CHECK-LABEL: lit.fn @"getThing()"
# CHECK-SAME: [mut *"__result__`"](?, %__result__:
fn getThing() -> MyStruct:
    # result slot parameter should get a different name to avoid conflict.
    # CHECK: lit.fn *"localTest()"
    # CHECK-SAME: [mut *"__result__`2x"](?, %__result___0[__result__]:
    fn localTest() -> MyStruct:
        return MyStruct()

    return localTest()


# CHECK-LABEL: lit.fn @"callThing()"
# CHECK-SAME: [mut *"__result__`"](?, %__result__:
fn callThing() -> MyStruct:
    return getThing()


##===----------------------------------------------------------------------===##
# Implicit Origin Parameters
##===----------------------------------------------------------------------===##


struct SomeType:
    pass


# COM: An implicit origin is passed into a struct parameter inside a trait
# COM: binding. Ensure this passes `-verify-parameters`.
# CHECK-LABEL: lit.fn @"implicit_origin_as_param
# CHECK-SAME: "__del__" : !lit.generator<[1]("self": !lit.ref<{{.*}}Match<:origin<0> *"arg`">, mut *[0,0]>
fn implicit_origin_as_param(
    arg: SomeType,
) -> Bound[Match[__origin_of(arg)]]:
    pass


struct Bound[T: AnyType]:
    pass


@value
struct Match[lt: __mlir_type.`!lit.origin<0>`]:
    pass


##===----------------------------------------------------------------------===##
# Struct field with type of recursive parameter
# https://github.com/modularml/modular/issues/28580
##===----------------------------------------------------------------------===##


trait BarTrait:
    pass


struct Bar[T: BarTrait]:
    fn __init__(out self):
        pass


struct BarSelf(BarTrait):
    var bar: Bar[Self]

    fn __init__(out self):
        # CHECK: [[V0:%.*]] = lit.ref.struct.ger %self
        # CHECK: lit.call{{.*}}__init__{{.*}}([[V0]])
        self.bar = Bar[Self]()


# CHECK-LABEL: lit.struct.decl @RegPassableInitSelfInit
@register_passable
struct RegPassableInitSelfInit:
    var a: Int

    # CHECK: lit.fn @"__init__
    # CHECK-SAME: () -> !RegPassableInitSelfInit
    fn __init__(out self):
        self.a = 42

    # CHECK: lit.fn @"__copyinit__
    # CHECK-SAME: -> !RegPassableInitSelfInit
    fn __copyinit__(out self, existing: Self):
        self.a = existing.a


# CHECK-LABEL: testRegPassableInitSelf
fn testRegPassableInitSelf():
    # CHECK-NEXT: %x = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: lit.ref.store [[TMP]], %x
    var x = RegPassableInitSelfInit()
    # CHECK-NEXT: %x2 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.immut %x
    # CHECK-NEXT: [[TMP2:%.*]] = lit.call {{.*}}__copyinit__{{.*}}([[TMP]])
    # CHECK-NEXT: lit.ref.store [[TMP2]], %x2
    var x2 = x

    # CHECK-NEXT: [[AP:%.*]] = lit.ref.struct.ger %x[a]
    # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant
    # CHECK-NEXT: lit.ref.store [[ONE:%.*]], [[AP]]
    x.a = 1


struct OverloadedKwArgs:
    var val: Int

    fn __init__(out self, single: Int):
        self.val = single

    fn __init__(out self, *, double: Int):
        self.val = double * 2

    fn __init__(out self, *, triple: Int):
        self.val = triple * 3

    fn __getitem__(self, idx: Int) -> Int:
        return self.val

    fn __getitem__(self, *, idx2: Int) -> Int:
        return self.val * 2

    fn __setitem__(mut self, idx: Int, val: Int):
        self.val = val

    fn __setitem__(mut self, val: Int, *, idx2: Int):
        self.val = val * 2

    fn overloaded_fn(mut self, x: Int, *, y: Int, z: Int):
        self.val = x + y + z

    fn overloaded_fn(mut self, x: Int, *, y2: Int, z: Int):
        self.val = x + y2 * 2 + z


# CHECK-LABEL: lit.fn @"testOverloadKwArgs
fn testOverloadKwArgs():
    # CHECK-NEXT: %x = lit.var.decl
    # CHECK-NEXT: %0 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %1 = lit.call @decls::@OverloadedKwArgs{{.*}}single
    var x = OverloadedKwArgs(1)

    # CHECK-NEXT: %2 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %3 = lit.call @decls::@OverloadedKwArgs{{.*}}single
    x = OverloadedKwArgs(single=1)

    # CHECK-NEXT: %4 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %5 = lit.call @decls::@OverloadedKwArgs{{.*}}double
    x = OverloadedKwArgs(double=1)

    # CHECK-NEXT: %6 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %7 = lit.call @decls::@OverloadedKwArgs{{.*}}triple
    x = OverloadedKwArgs(triple=1)

    # CHECK-NEXT: %8 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %9 = kgen.param.constant: !Int = <{42}>
    # CHECK-NEXT: %10 = lit.call @decls::@OverloadedKwArgs::@"__setitem__{{.*}}"idx"
    x[1] = 42

    # CHECK-NEXT: %11 = kgen.param.constant: !Int = <{42}>
    # CHECK-NEXT: %12 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %13 = lit.call @decls::@OverloadedKwArgs::@"__setitem__{{.*}}"idx2"
    x[idx2=1] = 42

    # CHECK-NEXT: %y = lit.var.decl
    # CHECK-NEXT: %14 = lit.ref.immut %x : <!OverloadedKwArgs, mut *"x`">
    # CHECK-NEXT: %15 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %16 = lit.call @decls::@OverloadedKwArgs::@"__getitem__{{.*}}idx
    # CHECK-NEXT: lit.ref.store %16, %y : <!Int, mut *"y`1">
    # CHECK-NEXT: %17 = lit.ref.immut %x : <!OverloadedKwArgs, mut *"x`">
    var y = x[1]

    # CHECK-NEXT: %18 = kgen.param.constant: !Int = <{1}
    # CHECK-NEXT: %19 = lit.call @decls::@OverloadedKwArgs::@"__getitem__{{.*}}idx2
    # CHECK-NEXT: lit.ref.store %19, %y : <!Int, mut *"y`1">
    y = x[idx2=1]

    # CHECK-NEXT: %z = lit.var.decl "z" var : !lit.ref<none, mut *"z`2">
    # CHECK-NEXT: %20 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %21 = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: %22 = kgen.param.constant: !Int = <{3}>
    # CHECK-NEXT: %23 = lit.call @decls::@OverloadedKwArgs{{.*}}"y"
    # CHECK-NEXT: lit.ref.store %23, %z : <none, mut *"z`2">
    var z = x.overloaded_fn(1, y=2, z=3)

    # CHECK-NEXT: %24 = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: %25 = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: %26 = kgen.param.constant: !Int = <{3}>
    # CHECK-NEXT: %27 = lit.call @decls::@OverloadedKwArgs{{.*}}"y2"
    # CHECK-NEXT: lit.ref.store %27, %z : <none, mut *"z`2">
    z = x.overloaded_fn(1, y2=2, z=3)


# Can't generate the constructors for a type wrapping !lit.ref
@value
struct MOCO1320[mut: Bool, //, origin: Origin[mut]]:
    alias _mlir_type = __mlir_type[
        `!lit.ref<`,
        Int,
        `, `,
        origin._mlir_origin,
        `>`,
    ]
    var _value: Self._mlir_type

    fn __init__(out self, *, x: Self._mlir_type):
        self._value = x

    fn __init__(out self, *, ref [origin]to: Int):
        self._value = __get_mvalue_as_litref(to)

struct StructWithParam[a: Int]: pass

# CHECK-LABEL: lit.fn @"autoparam_mangler_crash
fn autoparam_mangler_crash[*types: Int, constraints: StructWithParam](): pass
