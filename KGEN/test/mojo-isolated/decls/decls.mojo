# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s --kgen-print-inline-type-values | FileCheck %s


##===----------------------------------------------------------------------===##
# fn/def
##===----------------------------------------------------------------------===##


# Method overloading.
# CHECK-LABEL: lit.func @"testThing({{.*}}Int)"
fn testThing(a: Int) -> FloatDyn:
    return 1.0


# CHECK-LABEL: lit.func @"testThing({{.*}}Int,{{.*}}Int)"
fn testThing(a: Int, b: Int) -> Int:
    return 1


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
# https://github.com/modularml/mojo/issues/1443
fn variadic_trait_elt[T: Copyable](*xs: T):
    pass


# CHECK-LABEL: lit.func @"trait_pack
# CHECK-SAME: <{{.*}}, Ts:
# CHECK-SAME: %rest: !lit.struct<#VariadicPack <:i1 0, :lifetime<0> *"rest`1", :!lit.anytrait<!AnyType> !Copyable, :variadic<!Copyable> Ts>> borrow_in_mem|pack)
fn trait_pack[T: Copyable, *Ts: Copyable](first: T, *rest: *Ts):
    pass


# CHECK-LABEL: lit.func @"callOverload
fn callOverload(a: Int, pack: __mlir_type.`!kgen.pack<[index]>`):
    # CHECK: lit.call @decls::@"testThing({{.*}}Int)"(%a)
    _ = testThing(a)
    # CHECK: lit.call @decls::@"testThing({{.*}}Int,{{.*}}Int)"(%a, %a)
    _ = testThing(a, a)

    # CHECK: kgen.create_closure[!lit.signature<(!Int, |) -> !FloatDyn>:
    # CHECK-SAME: rebind(:!lit.signature<("a": !Int) -> !FloatDyn> @decls::@"testThing({{.*}}Int)")]()
    var float1: IntToFloat32Type = testThing

    # CHECK: kgen.create_closure[!lit.signature<(!Int, |) -> !FloatDyn>:
    # CHECK-SAME: rebind(:!lit.signature<("a": !Int) -> !FloatDyn> @decls::@"testThing({{.*}}Int)")]()
    # CHECK-NEXT: lit.ref.store %3, %float1
    float1 = testThing

    # CHECK: %4 = kgen.create_closure[!lit.signature<(!Int, |) -> !FloatDyn>:
    # CHECK-SAME: rebind(:!lit.signature<("a": !Int) -> !FloatDyn> @decls::@"testThing({{.*}}Int)")]()
    var float2: IntToFloat32Type = testThing

    # CHECK: lit.call @decls::@"takeIntToFloat32Param[fn({{.*}}Int, /) -> {{.*}}FloatDyn]()"<:
    # CHECK-SAME: !lit.signature<(!Int, |) -> !FloatDyn> rebind(:!lit.signature<("a": !Int) -> !FloatDyn> @decls::@"testThing{{.*}}")>()
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

    @always_inline("nodebug")
    fn __init__(inout self, _a: Int):
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


# CHECK-LABEL: lit.func @"callParametricOverload
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
    fn __init__(inout self):
        pass

    @staticmethod
    fn param_func[i: Int]():
        pass


fn take_variadic_struct[*Ts: AnyTrivialRegType](a: VariadicStruct[Ts]):
    pass


# CHECK-LABEL: lit.func @"variadic_params()"
fn variadic_params():
    # CHECK-NEXT: call {{.*}}param_func[{{.*}}Int]()"<:variadic<type> [!Int, !FloatDyn], :!Int {4}>
    VariadicStruct[Int, FloatDyn].param_func[4]()
    # CHECK: call {{.*}}take_variadic_struct{{.*}}<:variadic<type> [!Int, !FloatDyn]
    take_variadic_struct(VariadicStruct[Int, FloatDyn]())


# Test that pointers don't get confused with by-ref arguments.
# CHECK-LABEL: lit.func @"testPointerArgs{{.*}}(%ptr: !kgen.pointer<si32>) -> si32
fn testPointerArgs(ptr: __mlir_type.`!kgen.pointer<si32>`) -> __mlir_type.si32:
    # CHECK-NEXT: %0 = pop.load %ptr : !kgen.pointer<si32>
    return __mlir_op.`pop.load`[_type = __mlir_type.si32](ptr)


struct NoDebugInlineTest:
    # Two decorators stacked up
    @always_inline("nodebug")
    @staticmethod
    fn test():
        return


# CHECK-LABEL: lit.func @"testAlwaysInlineNoDebug
# CHECK-SAME: always_inline_no_debug
@always_inline("nodebug")
fn testAlwaysInlineNoDebug():
    pass


# CHECK-LABEL: lit.func @"testNoInline
# CHECK-SAME: no_inline
@no_inline
fn testNoInline():
    pass


# CHECK-LABEL: lit.func @"math{{.*}} always_inline_no_debug
@always_inline("nodebug")
fn math(a: __mlir_type.index, b: __mlir_type.index) -> __mlir_type.index:
    return __mlir_op.`index.add`(a, b)


# CHECK-LABEL: lit.func @"useIt
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


# CHECK-LABEL: lit.func @"callReturnParam
fn callReturnParam() -> __mlir_type.index:
    # CHECK-NEXT: %0 = lit.call @decls::@"returnParameter[__mlir_type.index]()"<3>()
    # CHECK-NEXT: return %0
    return returnParameter[Int(3).value]()


# CHECK: lit.func @"pleaseInline()"() -> index always_inline
@always_inline
fn pleaseInline() -> __mlir_type.index:
    return Int(1).value


# https://github.com/modularml/modular/issues/8500
struct AlwaysInlineByRef:
    @always_inline("nodebug")
    fn doByRef(inout self):
        pass


fn testInlineByRef(inout a: AlwaysInlineByRef):
    a.doByRef()


fn paramRefFunc[T: AnyTrivialRegType](x: T):
    pass


# CHECK-LABEL: lit.func @"orvalueInferType()"
fn orvalueInferType():
    fn func(x: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: call {{.*}}paramRefFunc{{.*}}<:type !lit.signature<("x": index) -> index>>
    paramRefFunc(func)


# CHECK-LABEL: lit.func @"kernel{{.*}}"<x:
# CHECK-SAME: LLVMMetadata = {nvvm.maxntid = {{.*}}#pop.array<x> : !pop.array<


@__llvm_metadata(
    `nvvm.maxntid`=__mlir_attr[`#pop.array<`, x, `> : !pop.array<1, `, Int, `>`]
)
fn kernel[x: Int]():
    pass


# https://github.com/modularml/mojo/issues/1152
# Allow mutable self argument when overloading operators using dunder methods
struct MutatingAdd:
    fn __add__(inout self, x: MutatingAdd):
        pass


# CHECK-LABEL: lit.func @"testMutatingAdd
fn testMutatingAdd(owned a: MutatingAdd, b: MutatingAdd):
    # CHECK-NEXT: lit.call {{.*}}__add__{{.*}}(%a, %b)
    a + b


##===----------------------------------------------------------------------===##
# Conventions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"ownedConventionMem
# CHECK-SAME: (%a: !lit.ref<!StructWithInit, mut {{.*}}> owned_in_mem,
# CHECK-SAME:  %b: !lit.ref<!StructWithInit, imm {{.*}}> borrow_in_mem)
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


# CHECK-LABEL: lit.func @"ownedConventionReg
# CHECK-SAME: (%a: !RPStructWithInit owned,
# CHECK-SAME:  %b: !RPStructWithInit,
# CHECK-SAME:  %triv: !RPStructWithInitTrivial)
fn ownedConventionReg(
    owned a: RPStructWithInit,
    b: RPStructWithInit,
    triv: RPStructWithInitTrivial,
):
    # CHECK: %a_0 = lit.var.decl "a" arg
    # CHECK: lit.ref.store %a, %a_0

    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a_0[x]
    # CHECK:  = lit.ref.load [[AX]]
    _ = a.x
    # CHECK: [[BY:%.*]] = lit.struct.extract %b[y]
    _ = b.y

    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a_0[x]
    # CHECK: [[ONE:%.*]]  = kgen.param.constant: !Int = <{1}>
    # CHECK: lit.ref.store [[ONE]], [[AX]]
    a.x = 1


struct BorrowStruct:
    fn testMethod(self):
        pass

    fn borrowedVarArgs(self, *x: BorrowStruct):
        pass


# CHECK-LABEL: callerFn
# CHECK-SAME: (%arg0: !lit.ref<{{.*}}> borrow_in_mem)
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
    fn __init__(inout self):
        pass


# CHECK-LABEL: lit.func @"named_result
# CHECK-SAME: %out: !lit.ref<!SomeResultType, {{.*}}> byref_result
# CHECK-SAME: namedResult = "out"
@__named_result(out)
fn named_result() -> SomeResultType:
    # CHECK-NEXT: call {{.*}}SomeResultType::@"__init__{{.*}}(%out)
    SomeResultType.__init__(out)
    # CHECK: lit.return %none
    return
    # CHECK-NEXT: lit.end_func


# CHECK-LABEL: lit.func @"named_result_return_expr
@__named_result(out)
fn named_result_return_expr() -> SomeResultType:
    # CHECK-NEXT: call {{.*}}SomeResultType::@"__init__{{.*}}(%out)
    return SomeResultType()


##===----------------------------------------------------------------------===##
# Default arguments and variadics.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"defaultArgument
# CHECK-SAME: %c: !Int = {5})
fn defaultArgument(a: Int, b: Int = 3, c: Int = 5) -> Int:
    return a + b


# CHECK-LABEL: lit.func @"callDefaultArgument
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


# CHECK-LABEL: lit.func @"defaultArgumentReferencesParameter
# CHECK-SAME: (%a: !Int = apply(:!lit.signature<("lhs": !Int, "rhs": !Int)
# CHECK-SAME: -> !Int> {{.*}}Int::@"__add__({{.*}}Int,{{.*}}Int)", {{.*}}p, {87}))
fn defaultArgumentReferencesParameter[p: Int](a: Int = p + 87) -> Int:
    return a


# CHECK-LABEL: lit.func @"defaultArgumentUntyped
# CHECK-SAME: borrow_in_mem = apply_result_slot({{.*}}object::@"__init__
def defaultArgumentUntyped(a=1):
    pass


struct MemoryType:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value


# CHECK-LABEL: lit.func @"defaultArgumentNonRegisterType
# CHECK-SAME: borrow_in_mem = apply_result_slot({{.*}}__init__
fn defaultArgumentNonRegisterType(a: MemoryType = 1):
    pass


# CHECK-LABEL: lit.func @"callNonRegisterDefaultArg
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


# CHECK: lit.func @"referencesDefaultArgumentFunction
fn referencesDefaultArgumentFunction():
    # CHECK: %f = lit.var.decl "f"
    # CHECK: lit.ref.store %0, %f
    var f = defaultArgument


# CHECK-LABEL: lit.struct.decl @Outer<X:
struct Outer[X: Int]:
    # CHECK: lit.func @"nested
    # CHECK-SAME: %x: !Int = X)
    fn nested(self, x: Int = X):
        pass


# CHECK-LABEL: lit.func @"variadics({{.*}}Int*)"(%a: !kgen.variadic<!Int> var)
fn variadics(*a: Int):
    # CHECK: lit.call {{.*}}VariadicList{{.*}}__init__
    pass


fn parameterizedVariadic[T: __mlir_type.`!kgen.type`](*args: T):
    pass


struct ParameterizedStruct[T: __mlir_type.`!kgen.type`]:
    fn __init__(inout self, *args: T):
        pass


struct VarArgsParameterizedStruct[*Is: Int]:
    fn __init__(inout self):
        pass


# CHECK-LABEL: lit.func @"callVariadic{{.*}}"<p: !Int>
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


# CHECK-LABEL: lit.func @"test_variadic_mem_only{{.*}}"<x: !MemStruct, y: !MemStruct>
fn test_variadic_mem_only[x: MemStruct, y: MemStruct]():
    # CHECK: lit.alias.decl {{.*}}: !Int = <apply(
    # CHECK-SAME: :!lit.signature<[1]("values": !kgen.variadic<!lit.ref<!MemStruct, imm #lit.lifetime>, borrow_in_mem> var) -> !Int> {{.*}}::@"variadic_mem_only({{.*}}::MemStruct*)"
    # CHECK-SAME: [store_to_mem(x), store_to_mem(y)]
    alias b = variadic_mem_only(x, y)


# CHECK-LABEL: lit.func @"implicit_return_obj
# CHECK-SAME: object{{.*}} byref_result
def implicit_return_obj(p: Bool):
    # CHECK: if
    if p:
        # CHECK: lit.call {{.*}}object::@"__init__{{.*}}%__result__
        # CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
        # CHECK: return [[FALSE]]
        return
    # CHECK: else
    else:
        # CHECK: lit.call {{.*}}object::@"__init__{{.*}}%__result__
        # CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
        # CHECK: return [[FALSE]]
        return 5
    # CHECK: lit.call {{.*}}object::@"__init__
    # CHECK: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK: return [[FALSE]]
    _ = 5


##===----------------------------------------------------------------------===##
# raises specifier.
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"defAlwaysRaises()"[{{.*}}](?, %__error__: {{.*}}, %__result__: {{.*}}) throws -> i1 attributes {isDef
def defAlwaysRaises() -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}{0}
    # CHECK: lit.ref.store [[RESULT]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return 0


# CHECK-LABEL: lit.func @"fnThatRaises()"{{.*}} throws -> i1
fn fnThatRaises() raises -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}{0}
    # CHECK-NEXT: lit.ref.store [[RESULT]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return 0


# CHECK-LABEL: lit.func @"raisesReturnsNone()"{{.*}} throws -> i1
fn raisesReturnsNone() raises:
    # CHECK-NEXT: %none = kgen.param.constant: none
    # CHECK-NEXT: lit.ref.store %none, %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    # CHECK-NEXT: lit.end_func
    pass


# COM: We can return an variant of error and index in a non-throwing function.
# CHECK-LABEL: lit.func @"raisesReturnsVariant()"() -> !kgen.variant<!Error, index>
fn raisesReturnsVariant() -> __mlir_type[`!kgen.variant<`, Error, `, index>`]:
    return __mlir_op.`kgen.variant.create`[
        _type = __mlir_type[`!kgen.variant<`, Error, `, index>`],
        index = Int(1).value,
    ](Int(1).value)


# CHECK-LABEL: lit.func @"raise_and_return{{.*}} throws -> i1
fn raise_and_return(a: Error) raises -> Error:
    # COM: True result indicates an error.
    # CHECK: [[ERR:%.*]] = kgen.param.materialize: !Error
    # CHECK-NEXT: lit.ref.store [[ERR]], %__result__
    # CHECK-NEXT: [[FALSE:%.*]] = kgen.param.constant: i1 = <0>
    # CHECK-NEXT: lit.return [[FALSE]]
    return Error {}


@value
@register_passable("trivial")
struct RaisingGetterSetter:
    fn __getitem__(self, i: Int) raises -> FloatDyn:
        return 1.0

    fn __setitem__(inout self, i: Int, v: FloatDyn) raises:
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

    # CHECK: lit.func @"__init__(decls::StructWithInit=&,{{.*}}Int)"
    # CHECK-SAME: (%self: !lit.ref<!StructWithInit, mut {{.*}}> init_self,
    fn __init__(inout self, a: Int):
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
    # CHECK: lit.func @"__init__
    # CHECK-SAME: (%self: !lit.ref<!StructWithInit, mut {{.*}}> init_self,
    fn __init__(inout self, a: Int, b: Int):
        # CHECK: hlcf.elif
        if a == b:
            # CHECK:  lit.call {{.*}}__init__{{.*}}(%self, %a)
            self = StructWithInit(a)
        else:
            # CHECK: [[XP:%.*]] = lit.ref.struct.ger %self[x]
            # CHECK: lit.ref.store %a, [[XP]]
            # CHECK: [[YP:%.*]] = lit.ref.struct.ger %self[y]
            # CHECK: lit.ref.store %b, [[YP]]
            self.x = a
            self.y = b

    fn __init__(inout self):
        self = Self(0)


# CHECK-LABEL: lit.struct.decl @StructExample
@register_passable
struct StructExample:
    fn __copyinit__(self) -> Self:
        return Self()

    fn __init__(inout self):
        pass

    # CHECK: lit.func @"maybe_static({{.*}}Int)"(%x: !Int) {{.*}}isStatic
    @staticmethod
    fn maybe_static(x: Int):
        # CHECK: %0 = {{.*}}{4}
        # CHECK: lit.call @decls::@StructExample::@"maybe_static{{.*}}"(%0)
        StructExample.maybe_static(4)
        pass

    # This isn't static.
    # CHECK: lit.func @"maybe_static
    fn maybe_static(self, x: EmptyStruct):
        # CHECK: %0 = {{.*}}{4}
        # CHECK: lit.call @decls::@StructExample::@"maybe_static{{.*}}"(%0)
        StructExample.maybe_static(4)
        pass

    # CHECK: lit.func @"mutatingMethod{{.*}}(%self: !lit.ref<!StructExample, mut {{.*}}> inout) -> !kgen.none
    fn mutatingMethod(inout self):
        pass


# CHECK-LABEL: lit.func @"callMaybeStatic{{.*}}(%a: !Int, %b: !EmptyStruct)
fn callMaybeStatic(a: Int, b: EmptyStruct):
    # CHECK-NEXT: lit.call @decls::@StructExample::@"maybe_static{{.*}}(%a)
    StructExample.maybe_static(a)

    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}@StructExample::@"__init__{{.*}}([[ANONSE]])
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load [[ANONSE]]
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}([[TMP]], %b)
    StructExample.maybe_static(StructExample(), b)

    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}@"__init__{{.*}}([[ANONSE]])
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}(%a)
    StructExample().maybe_static(a)

    # CHECK-NEXT: [[ANONSE:%.*]] = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}@StructExample::@"__init__{{.*}}([[ANONSE]])
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load [[ANONSE]]
    # CHECK-NEXT: lit.call {{.*}}@"maybe_static{{.*}}([[TMP]], %b)
    StructExample().maybe_static(b)


# CHECK-LABEL: lit.struct.decl @DelegatingInitMem
# Issue #12042
struct DelegatingInitMem:
    var value: Int

    # CHECK: lit.func @"__init__{{.*}}(%self
    fn __init__(inout self, value: Bool):
        # CHECK: lit.call @{{.*}}__init__{{.*}}(%self, %0)
        self.__init__(42)

    fn __init__(inout self, value: Int):
        self.value = value


# External issue #260
fn nameOutsideStruct(x: Int, y: Int):
    pass


struct ShadowsOuterName:
    fn nameOutsideStruct(self: Self):
        nameOutsideStruct(1, 2)


##===----------------------------------------------------------------------===##
# Struct @value decorator
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.struct.decl @ValueMem(!AnyType, !Copyable, !Movable)
# CHECK: move :!lit.signature<[2]({{.*}} init_self, {{.*}} owned_in_mem, |) {{.*}}ValueMem::@"__moveinit__
@value
struct ValueMem:
    var a: Int  # Trivial
    var b: StructExample  # Copy ctor


# CHECK: lit.func @"__moveinit__(
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> init_self,
# CHECK-SAME:  %other: !lit.ref<!ValueMem, mut {{.*}}> owned_in_mem, |)
# CHECK-SAME: -> !kgen.none always_inline_no_debug attributes
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %other[a]
# CHECK-NEXT: %2 = lit.load.consume %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %other[b]
# CHECK-NEXT: %5 = lit.load.consume %4
# CHECK-NEXT: lit.ref.store %5, %3

# CHECK: lit.func @"__copyinit__(
# CHECK-SAME:  %self: !lit.ref<!ValueMem, mut {{.*}}> init_self,
# CHECK-SAME:  %other: !lit.ref<!ValueMem, imm {{.*}}> borrow_in_mem, |)
# CHECK-SAME: -> !kgen.none always_inline_no_debug attributes
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %other[a]
# CHECK-NEXT: %2 = lit.ref.load %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %other[b]
# CHECK-NEXT: %5 = lit.ref.load %4
# CHECK-NEXT: %6 = lit.call {{.*}}__copyinit__{{.*}}(%5)
# CHECK-NEXT: lit.ref.store %6, %3

# CHECK: lit.func @"__init__(
# CHECK-SAME:  %[[SELF:.*]][*""]: !lit.ref<!ValueMem, mut {{.*}}> init_self,
# CHECK-SAME:  %a: !Int,
# CHECK-SAME:  %b: !StructExample
# CHECK-SAME: ) -> !kgen.none always_inline_no_debug attributes {isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
# CHECK-NEXT: %[[PA:.*]] = lit.ref.struct.ger %[[SELF]][a]
# CHECK-NEXT: lit.ref.store %a, %[[PA]]
# CHECK-NEXT: %[[PB:.*]] = lit.ref.struct.ger %[[SELF]][b]
# CHECK-NEXT: lit.ref.store %b, %[[PB]]


# CHECK-LABEL: lit.struct.decl @ValueMemHasCopy(!AnyType, !Copyable, !Movable)
@value
struct ValueMemHasCopy:
    var a: Int
    var b: StructExample

    fn __copyinit__(inout self, other: Self):
        self.a = other.a
        self.b = other.b


# CHECK-LABEL: lit.struct.decl @ValueMemHasMove(!AnyType, !Copyable, !Movable)
@value
struct ValueMemHasMove:
    var a: Int
    var b: StructExample

    fn __moveinit__(inout self, owned other: Self):
        self.a = other.a
        self.b = other.b


# CHECK-LABEL: lit.struct.decl @ValueRegTrivial
# CHECK-SAME: (!AnyType, !Copyable, !Movable) register_passable_trivial

# CHECK: lit.func @"__copyinit__{{.*}}_thunk"[{{.*}}](%0[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> init_self, %1[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> borrow_in_mem, |) -> !kgen.none always_inline_no_debug
# CHECK-NEXT: [[V0:%.*]] = lit.ref.load %1 : <!ValueRegTrivial
# CHECK-NEXT: lit.ref.store [[V0]], %0
# CHECK-NEXT: %none = kgen.param.constant: none = <#kgen.none>
# CHECK-NEXT: kgen.return %none : !kgen.none

# CHECK: lit.func @"__moveinit__{{.*}}_thunk"[{{.*}}](%0[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> init_self, %1[*""]: !lit.ref<!ValueRegTrivial, {{.*}}> owned_in_mem, |) -> !kgen.none
# CHECK-NEXT: [[V0:%.*]] = lit.load.consume %1
# CHECK-NEXT: lit.ref.store [[V0]], %0
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


# CHECK: lit.func @"__copyinit__
# CHECK-SAME: (%other: !ValueReg, |)
# CHECK-SAME:  -> !ValueReg
# CHECK-SAME: attributes {{.*}}specialFnKind = 6 : i8
# CHECK-NEXT: %0 = lit.struct.extract %other[a]
# CHECK-NEXT: %1 = lit.struct.extract %other[b]
# CHECK-NEXT: %2 = lit.call {{.*}}__copyinit__{{.*}}(%1)
# CHECK-NEXT: %3 = lit.struct.create(a=%0, b=%2)
# CHECK-NEXT: lit.return %3

# CHECK: lit.func @"__init__(
# CHECK-SAME:  %a: !Int,
# CHECK-SAME:  %b: !StructExample
# CHECK-SAME: ) -> !ValueReg
# CHECK-NEXT: %0 = lit.struct.create(a=%a, b=%b)
# CHECK-NEXT: lit.return %0
# CHECK-NEXT: lit.end_func


# COM: Ensure that "self" is a valid field name.
# CHECK-LABEL: lit.struct.decl @Foo(!AnyType, !Copyable, !Movable) attributes
@value
struct Foo:
    var a: Int
    var self: Int


# CHECK: lit.func @"__init__{{.*}}(%[[SELFARG:.*]][*""]: !lit.ref<!Foo, mut {{.*}}> init_self, |, %a: !Int, %self: !Int)


# CHECK-LABEL: lit.struct.decl @ParamVarArg<I: variadic<!Int> var>
@value
@register_passable("trivial")
struct ParamVarArg[*I: Int]:
    pass


# CHECK-LABEL: lit.struct.decl @TraitMember
@value
struct TraitMember[T: Copyable]:
    var value: T
    # CHECK: lit.func @"__moveinit__
    # CHECK: call{{.*}}__copyinit__
    # CHECK: lit.func @"__copyinit__
    # CHECK: call{{.*}}__copyinit__


# CHECK: lit.func @"notSynthetic{{.*}}(%self: !lit.ref<!NotSynthetic, imm {{.*}}> borrow_in_mem) -> !kgen.none attributes {sourceName = "notSynthetic", specialFnKind = 0 : i8}
# CHECK: lit.func @"__moveinit__{{.*}}isSynthetic
# CHECK: lit.func @"__copyinit__{{.*}}isSynthetic
# CHECK: lit.func @"__init__{{.*}}isSynthetic
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

    # CHECK: lit.func @"__init__(decls::VarArgInit=&,decls::ValueMem*)"{{.*}}({{.*}}: !kgen.variadic<!lit.ref<!ValueMem, imm {{.*}}>, borrow_in_mem> var)
    # The argument is intentionally memory-only.
    fn __init__(inout self, *values: ValueMem):
        self.a = 42

    # CHECK: lit.func @"__init__({{.*}}Int)"{{.*}}({{.*}}, %a: !Int)


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

    # CHECK-LABEL: lit.func @"__init__{{.*}} throws
    fn __init__(inout self, /, x: Int) raises:
        pass


##===----------------------------------------------------------------------===##
# async/await
##===----------------------------------------------------------------------===##

@register_passable("trivial")
struct Container[T : AnyType]:
    alias _mlir_type = __mlir_type[
        `!kgen.pointer<`,
        T,
        `>`,
    ]
    var address: Self._mlir_type
    fn __init__() -> Self:
        return Self {
            address: __mlir_attr[`#interp.pointer<0> : `, Self._mlir_type]
        }

async fn load(server_ptr: Container[__mlir_type.index]):
    pass

# CHECK-LABEL: lit.func @"awaitSomething()"
async fn awaitSomething():
    var ptr = Container[__mlir_type.index]()
    # CHECK: [[CORO:%.*]] = lit.call @{{.*}}@Coroutine::@"__init__{{.*}}"<:!AnyType [{{.*}}], :lifetime.set {}>(%{{.*}}) :
    # CHECK-SAME: !lit.signature<("{{.*}}": !co.routine) -> !lit.struct<#Coroutine <:!AnyType [{{.*}}], :lifetime.set {}>>>
    await load(ptr)

# CHECK-LABEL: lit.func @"coroutine
# CHECK-SAME: [mut [[LT:.*]]](?, %__result__: !lit.ref<!Int, mut [[LT]]> byref_result) async -> !kgen.none
async fn coroutine() -> Int:
    # CHECK: lit.ref.store %0, %__result__
    # CHECK: lit.return %none
    return 0


# CHECK-LABEL: lit.struct.decl @StructWithAsync
struct StructWithAsync:
    # CHECK-LABEL: lit.func @"do_something{{.*}}({{.*}}) async
    async fn do_something(self: StructWithAsync):
        # CHECK-NEXT: %[[CORO:.*]] = lit.async.call[!lit.signature<[1](?, "__result__": !lit.ref<!Int, mut *[0,0]> byref_result) async -> !kgen.none>: @decls::@"coroutine()"][mut #lit.lifetime]()
        # CHECK-NEXT: %[[COROUTINE:.*]] = lit.call {{.*}}@Coroutine::@"__init__{{.*}}<:!AnyType [!Int, {{.*}}], :lifetime.set {}>(%[[CORO]])
        _ = coroutine()


# CHECK-LABEL: lit.func @"call_struct_async
# CHECK-SAME: [imm [[LT:.*]], mut {{.*}}]{{.*}}) async -> !kgen.none
async fn call_struct_async(f: StructWithAsync):
    # CHECK-NEXT: lit.async.call[!lit.signature<[2]({{.*}}, "__result__":{{.*}}) async -> !kgen.none>: @{{.*}}][imm [[LT]], mut #lit.lifetime](%f)
    _ = f.do_something()


struct Awaitable:
    fn __init__(inout self):
        pass

    fn __await__(inout self) -> Int:
        return 0


# CHECK-LABEL: lit.func @"awaitable()"
fn awaitable() -> Int:
    # CHECK: call {{.*}}@Awaitable::@"__await__{{.*}}(%aw)
    var aw = Awaitable()
    return await aw


# COM: https://github.com/modularml/mojo/issues/951
@always_inline
async fn inline_async() -> Int:
    return 0


# CHECK-LABEL: lit.func @"use_inline_async()"
async fn use_inline_async() -> Int:
    # CHECK: [[ASYNC_RESULT:%.*]] = lit.async.call{{.*}}inline_async
    # CHECK: [[CORO:%.*]] = lit.call {{.*}}Coroutine{{.*}}__init__{{.*}}[[ASYNC_RESULT]]
    # CHECK: [[RESULT:%.*]] = lit.call {{.*}}Coroutine{{.*}}__await__{{.*}}([[CORO]], %__result__)
    return await inline_async()


async fn capture_byref(inout x: Awaitable, y: Awaitable):
    pass


@value
@register_passable
struct LifetimeAccess[lifetime: __mlir_type.`!lit.lifetime<1>`]:
    pass


async fn lifetime_access(owned x: LifetimeAccess[_]):
    pass


# CHECK-LABEL: lit.func @"coroutine_lifetimes
fn coroutine_lifetimes():
    # CHECK: var.decl "x" var : {{.*}}mut [[X_LT:.*]]>
    var x: Awaitable
    # CHECK: var.decl "y" var : {{.*}}mut [[Y_LT:.*]]>
    var y: Awaitable
    # CHECK: [[Y_IMM:%.*]] = lit.ref.immut %y
    # CHECK: [[CORO:%.*]] = lit.async.call[!lit.signature<[3]("x": !lit.ref<!Awaitable, mut *[0,0]> inout, "y": !lit.ref<!Awaitable, imm *[0,1]> borrow_in_mem, ?, "__result__": !lit.ref<none, mut *[0,2]> byref_result) async -> !kgen.none>
    # CHECK-SAME: [mut [[X_LT]], muttoimm [[Y_LT]], mut #lit.lifetime](%x, [[Y_IMM]])
    # CHECK: [[CORO_VAL:%.*]] = lit.call {{.*}}Coroutine::@"__init__{{.*}}<:!AnyType [none, {{.*}}], :lifetime.set {mut [[X_LT]], mut [[Y_LT]]}>([[CORO]])
    # CHECK: store [[CORO_VAL]], %coro : <{{.*}}Coroutine<:!AnyType [none, {{.*}}], :lifetime.set {mut [[X_LT]], mut [[Y_LT]]}>
    var coro = capture_byref(x, y)

    # CHECK: lit.async.call[!lit.signature<[1]("x": !lit.struct<#LifetimeAccess <:lifetime<1> [[Y_LT]]>> owned, {{.*}}) async -> !kgen.none>: {{.*}}lifetime_access{{.*}}<:lifetime<1> [[Y_LT]]>]
    # CHECK: Coroutine<:!AnyType [none, {{.*}}], :lifetime.set {mut [[Y_LT]]}>
    var access = lifetime_access(LifetimeAccess[__lifetime_of(y)]())


# CHECK-LABEL: lit.func @"mem_result{{.*}}(?, %__result__: !lit.ref<!Awaitable, {{.*}}> byref_result) async -> !kgen.none
async fn mem_result() -> Awaitable:
    # CHECK: [[CORO:%.*]] = lit.async.call[{{.*}}mem_result()"][mut #lit.lifetime]()
    # CHECK-NEXT: lit.call {{.*}}@Coroutine::@"__init__{{.*}}([[CORO]])
    var coro = mem_result()


# CHECK-LABEL: lit.func @"mem_raises{{.*}}(?, %__error__: !lit.ref<!Error, {{.*}}> byref_error, %__result__: !lit.ref<!Int, {{.*}}> byref_result) throws|async -> i1
async fn mem_raises() raises -> Int:
    # CHECK: [[CORO:%.*]] = lit.async.call[{{.*}}mem_raises()"][mut #lit.lifetime, mut #lit.lifetime]()
    # CHECK-NEXT: lit.call {{.*}}@RaisingCoroutine::@"__init__{{.*}}([[CORO]])
    var coro = mem_raises()


##===----------------------------------------------------------------------===##
# Nested Functions
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.func @"topLevelFunction()"
fn topLevelFunction() -> Int:
    var a = 0

    # CHECK: lit.func *"nestedFunction()"
    @parameter
    fn nestedFunction() -> Int:
        # CHECK-NEXT: lit.ref.load %a
        return a

    # CHECK: lit.alias.decl *"b{{.*}}": !lit.signature<() capturing -> !Int> = <*"nestedFunction()">
    alias b = nestedFunction
    # CHECK: call[!lit.signature<() capturing -> !Int>: *"nestedFunction()"]()
    return nestedFunction()


# CHECK-LABEL: lit.struct.decl @SomeStruct
struct SomeStruct:
    # CHECK-LABEL: @"someMethod({{.*}})"
    fn someMethod(self) -> Int:
        var a = 0

        # CHECK: lit.func *"nestedFunction()"
        @parameter
        fn nestedFunction() -> Int:
            # CHECK-NEXT: lit.ref.load %a
            return a

        # CHECK: lit.alias.decl *"b{{.*}}": !lit.signature<() capturing -> !Int> = <*"nestedFunction()">
        alias b = nestedFunction
        # CHECK: call[!lit.signature<() capturing -> !Int>: *"nestedFunction()"]()
        return nestedFunction()


# CHECK-LABEL: lit.func @"closureParameter[fn() capturing -> __mlir_type.index]()"
# CHECK-SAME: capturing ->
fn closureParameter[func: fn () capturing -> __mlir_type.index]():
    pass


# CHECK-LABEL: lit.func @"topLevelParamFn[__mlir_type.index]()"<a_param>
fn topLevelParamFn[a_param: __mlir_type.index]():
    # CHECK: lit.func *"nestedFunction[__mlir_type.index]()"<b_param>
    fn nestedFunction[b_param: __mlir_type.index]():
        return

    # CHECK: lit.alias.decl *"thinref{{.*}}": !lit.signature<<"b_param": index>() -> !kgen.none> = <*"nestedFunction[__mlir_type.index]()">
    alias thinref = nestedFunction
    # CHECK: call[{{.*}}: bind_signature(:!lit.signature<<"b_param": index>() -> !kgen.none> *"nestedFunction[__mlir_type.index]()", 2)]()
    nestedFunction[Int(2).value]()

    var value = 0

    @__copy_capture(value)
    @parameter
    fn capturingNestedFunction() -> Int:
        return value

    # CHECK: lit.alias.decl *"fatRef{{.*}}": !lit.signature<() capturing -> !Int> = <*"capturingNestedFunction()">
    alias fatRef = capturingNestedFunction


struct SomeParamStruct[c_param: Int]:
    # CHECK-LABEL: lit.func @"topLevelParamFn{{.*}}<a_param: !Int>
    fn topLevelParamFn[a_param: Int](self):
        # CHECK: lit.func *"nestedFunction{{.*}}"<b_param: !Int>
        fn nestedFunction[b_param: Int]():
            return

        # CHECK: lit.alias.decl *"reff{{.*}}": !lit.signature<<"b_param": !Int>() -> !kgen.none> = <*"nestedFunction[{{.*}}Int]()">
        alias reff = nestedFunction
        # CHECK: call[{{.*}}: bind_signature(:!lit.signature<<"b_param": !Int>() -> !kgen.none> *"nestedFunction[{{.*}}Int]()", {{.*}}2{{.*}})]()
        nestedFunction[2]()


##===----------------------------------------------------------------------===##
# Exported Functions
##===----------------------------------------------------------------------===##


@export("my_named_export", ABI="C")
# CHECK: lit.func export C @"export_me()"
# CHECK-SAME: linkageName = "my_named_export"
def export_me() -> None:
    ...


@export
# CHECK: lit.func export @"not_c_exported()"
fn not_c_exported():
    pass


struct Thing:
    # CHECK: lit.func export @"member
    @export
    fn member(self):
        pass


##===----------------------------------------------------------------------===##
# Decorators
##===----------------------------------------------------------------------===##


fn decorator():
    return


fn decorator_arg(a: Int):
    return


# CHECK-LABEL: lit.func @"decorated_fn()"
# CHECK-NEXT: decorators <:!lit.signature<() -> !kgen.none> @{{.*}}::@"decorator()">
@decorator
fn decorated_fn():
    pass


# CHECK-LABEL: lit.struct.decl @DecoratedStruct
# CHECK: decorators <:none apply({{.*}}decorator_arg{{.*}}, {2}
@decorator_arg(2)
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


# CHECK-LABEL: lit.func @"deprecated_func
# CHECK-SAME: deprecationWarning = "func"
@deprecated("func")
fn deprecated_func():
    pass


# CHECK-LABEL: lit.trait.decl @DeprecatedTrait
# CHECK-SAME: deprecationWarning = "trait"
@deprecated("trait")
trait DeprecatedTrait:
    pass


# CHECK-LABEL: lit.globalvar.decl @deprecated_global
# CHECK-SAME: deprecationWarning = "global"
@deprecated("global")
var deprecated_global = 1


# CHECK-LABEL: lit.alias.decl
# CHECK-SAME: deprecationWarning = "alias"
@deprecated("alias")
alias deprecated_alias = 1


##===----------------------------------------------------------------------===##
# Implicit lifetimes for result slots.
##===----------------------------------------------------------------------===##


struct MyStruct:
    fn __init__(inout self):
        pass


# CHECK-LABEL: lit.func @"getThing()"
# CHECK-SAME: [mut *"__result__`"](?, %__result__:
fn getThing() -> MyStruct:
    # result slot parameter should get a different name to avoid conflict.
    # CHECK: lit.func *"localTest()"
    # CHECK-SAME: [mut *"__result__`2x"](?, %__result___0[__result__]:
    fn localTest() -> MyStruct:
        return MyStruct()

    return localTest()


# CHECK-LABEL: lit.func @"callThing()"
# CHECK-SAME: [mut *"__result__`"](?, %__result__:
fn callThing() -> MyStruct:
    return getThing()


##===----------------------------------------------------------------------===##
# Implicit Lifetime Parameters
##===----------------------------------------------------------------------===##


struct SomeType:
    pass


# COM: An implicit lifetime is passed into a struct parameter inside a trait
# COM: binding. Ensure this passes `-verify-parameters`.
# CHECK-LABEL: lit.func @"implicit_lifetime_as_param
# CHECK-SAME: "__del__" : !lit.signature<[1]("self": !lit.ref<{{.*}}Match<:lifetime<0> *"arg`">, mut *[0,0]>
fn implicit_lifetime_as_param(
    arg: SomeType,
) -> Bound[Match[__lifetime_of(arg)]]:
    pass


struct Bound[T: AnyType]:
    pass


@value
struct Match[lt: __mlir_type.`!lit.lifetime<0>`]:
    pass


##===----------------------------------------------------------------------===##
# Struct field with type of recursive parameter
# https://github.com/modularml/modular/issues/28580
##===----------------------------------------------------------------------===##


trait BarTrait:
    pass


struct Bar[T: BarTrait]:
    fn __init__(inout self: Self):
        pass


struct BarSelf(BarTrait):
    var bar: Bar[Self]

    fn __init__(inout self: Self):
        # CHECK: [[V0:%.*]] = lit.ref.struct.ger %self
        # CHECK: lit.call{{.*}}__init__{{.*}}([[V0]])
        self.bar = Bar[Self]()


# CHECK-LABEL: lit.struct.decl @RegPassableInitSelfInit
@register_passable
struct RegPassableInitSelfInit:
    var a: Int

    # CHECK: lit.func @"__init__
    # CHECK-SAME: (%self: !lit.ref<!RegPassableInitSelfInit, mut {{.*}}> init_self,
    fn __init__(inout self):
        self.a = 42

    # CHECK: lit.func @"__copyinit__
    # CHECK-SAME: (%self: !lit.ref<!RegPassableInitSelfInit, mut {{.*}}> init_self,
    fn __copyinit__(inout self, existing: Self):
        self.a = existing.a


# CHECK-LABEL: testRegPassableInitSelf
fn testRegPassableInitSelf():
    # CHECK-NEXT: %x = lit.var.decl
    # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}(%x)
    var x = RegPassableInitSelfInit()
    # CHECK-NEXT: %x2 = lit.var.decl
    # CHECK-NEXT: [[TMP:%.*]] = lit.ref.load %x
    # CHECK-NEXT: lit.call {{.*}}__copyinit__{{.*}}(%x2, [[TMP]])
    var x2 = x

    # CHECK-NEXT: [[AP:%.*]] = lit.ref.struct.ger %x[a]
    # CHECK-NEXT: [[ONE:%.*]] = kgen.param.constant
    # CHECK-NEXT: lit.ref.store [[ONE:%.*]], [[AP]]
    x.a = 1
