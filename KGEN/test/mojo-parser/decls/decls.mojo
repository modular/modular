# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate %s -import-mojo | FileCheck %s


##===----------------------------------------------------------------------===##
# fn/def
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"empty_def()"() -> !kgen.none
# CHECK: lit.end_func
fn empty_def():
    pass

# CHECK-LABEL: lit.func @"slash
# CHECK-SAME: (%a[a]: !Int borrow, |, %b[b]: !Int borrow)
fn slash(a: Int, /, b: Int):
    pass

# CHECK-LABEL: lit.func @"slashLast
# CHECK-SAME: (%a[a]: !Int borrow, |, %b[b]: !Int borrow)
fn slashLast(a: Int, /, b: Int):
    pass

# Method overloading.
# CHECK-LABEL: lit.func @"testThing({{.*}}$int::Int)"
fn testThing(a: Int) -> Float32:
    return 1.0


# CHECK-LABEL: lit.func @"testThing({{.*}}$int::Int,{{.*}}$int::Int)"
fn testThing(a: Int, b: Int) -> Int:
    return 1


alias IntToFloat32Type = fn(Int) -> Float32


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


fn packOverload[*Ts: __mlir_type.`!kgen.anyregtype`](*a: *Ts):
    pass


fn packOverload():
    pass


fn directly_pass_pack(pack: __mlir_type.`!kgen.pack<[index]>`):
    pass


# CHECK-LABEL: lit.func @"trait_pack
# CHECK-SAME: <{{.*}}, [[TS:.*]][Ts]:
# CHECK-SAME: !kgen.pack<:variadic<trait<{{.*}}Intable>> [[TS]]> borrow
fn trait_pack[T: Intable, *Ts: Intable](first: T, *rest: *Ts):
    pass


# CHECK-LABEL: lit.func @"callOverload
fn callOverload(a: Int, pack: __mlir_type.`!kgen.pack<[index]>`):
    # CHECK: lit.call @"$decls"::@"testThing({{.*}}$int::Int)"(%a)
    _ = testThing(a)
    # CHECK: lit.call @"$decls"::@"testThing({{.*}}$int::Int,{{.*}}$int::Int)"(%a, %a)
    _ = testThing(a, a)

    # CHECK: kgen.create_closure[!lit.signature<(!Int borrow, |) -> !kgen.declref<{{.*}}>>:
    # CHECK-SAME: rebind(:!lit.signature<("a": !Int borrow) -> !kgen.declref<{{.*}}>> @"$decls"::@"testThing({{.*}}$int::Int)")]()
    var float1: IntToFloat32Type = testThing

    # CHECK: kgen.create_closure[!lit.signature<(!Int borrow, |) -> !kgen.declref<{{.*}}>>:
    # CHECK-SAME: rebind(:!lit.signature<("a": !Int borrow) -> !kgen.declref<{{.*}}>> @"$decls"::@"testThing({{.*}}$int::Int)")]()
    # CHECK-NEXT: lit.ref.store %3, %float1
    float1 = testThing

    # CHECK: %4 = kgen.create_closure[!lit.signature<(!Int borrow, |) -> !kgen.declref<{{.*}}>>:
    # CHECK-SAME: rebind(:!lit.signature<("a": !Int borrow) -> !kgen.declref<{{.*}}>> @"$decls"::@"testThing({{.*}}$int::Int)")]()
    let float2: IntToFloat32Type = testThing

    # CHECK: lit.call @"$decls"::@"takeIntToFloat32Param[fn({{.*}}::Int, /) -> $builtin::$simd::SIMD[{f32}, {1}]]()"<:
    # CHECK-SAME: !lit.signature<(!Int borrow, |) -> !kgen.declref<{{.*}}SIMD{{.*}}f32{{.*}}>> rebind(:!lit.signature<("a": !Int borrow) -> !kgen.declref<{{.*}}>> @"$decls"::@"testThing{{.*}}")>()
    takeIntToFloat32Param[testThing]()

    # Issue #10036.  This should call the exact match, consider the varargs match
    # less specific.
    # CHECK: lit.call @"$decls"::@"varargOverload({{.*}}$int::Int)"(%{{.*}})
    varargOverload(2)

    # CHECK:  lit.call @"$decls"::@"varargOverload()"()
    varargOverload()

    # Expect packs to behave similarly to varargs.
    # CHECK: %[[IDX3:.*]] = {{.*}}constant{{.*}} 3
    # CHECK: lit.call @"$decls"::@"packOverload({{.*}}$int::Int)"(%[[IDX3]])
    packOverload(3)
    # CHECK:  lit.call @"$decls"::@"packOverload()"()
    packOverload()

    # CHECK-NOT: pack.create
    # CHECK: call {{.*}}directly_pass_pack{{.*}}(%pack)
    directly_pass_pack(pack)

    # CHECK: call {{.*}}trait_pack
    # CHECK-SAME: [!Int, {"__int__"
    trait_pack(1, 2, 3)


@register_passable("trivial")
struct MyInt:
    var value: Int

    @always_inline("nodebug")
    fn __init__(_a: Int) -> Self:
        return Self {value: _a}


fn paramOverload[x: Int]():
    pass


fn paramOverload[x: Int, y: Int]():
    pass


fn paramOverload[*x: Int]():
    pass


fn paramOverload(y: Int):
    pass


fn paramOverload[x: Int, T: AnyRegType](y: T):
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
    # CHECK: lit.call @"$decls"::@"paramOverload[{{.*}}$int::Int]()"
    paramOverload[a]()

    # CHECK: lit.call @"$decls"::@"paramOverload[{{.*}}$int::Int,{{.*}}$int::Int]()"
    paramOverload[a, b]()

    # CHECK: lit.call @"$decls"::@"paramOverload[{{.*}}variadic<{{.*}}Int{{.*}}>]()"
    paramOverload[a, b, c]()

    # CHECK: lit.call @"$decls"::@"paramOverload({{.*}}$int::Int)"
    paramOverload(x)

    # CHECK: lit.call @"$decls"::@"paramOverload[{{.*}}$int::Int,AnyRegType]($1)"
    paramOverload[a](x)

    # CHECK: lit.call @"$decls"::@"paramOverload[{{.*}}variadic<{{.*}}Int{{.*}}>]({{.*}}$int::Int)"
    paramOverload[a, b](x)

    # CHECK: lit.call @"$decls"::@"paramOverload2[{{.*}}variadic<{{.*}}Int{{.*}}>]()"
    paramOverload2[a]()

    # CHECK: lit.call @"$decls"::@"paramOverload2[{{.*}}variadic<{{.*}}Int{{.*}}>]()"
    paramOverload2[a, b]()

    # CHECK: lit.call @"$decls"::@"paramOverload2[$decls::MyInt]()"
    paramOverload2[MyInt(a)]()

    # CHECK: lit.call @"$decls"::@"paramOverload2[$decls::MyInt,$decls::MyInt]()"
    paramOverload2[MyInt(a), b]()

    # CHECK: lit.call @"$decls"::@"paramOverload2[{{.*}}variadic<{{.*}}MyInt{{.*}}>]()"
    paramOverload2[MyInt(a), b, c]()


# Test overloading precedence in the presence of static methods.
struct MyStruct:
    fn __init__(inout self): pass

    fn foo(inout self): pass

    @staticmethod
    fn foo(): pass

# CHECK-LABEL: lit.func @"test_static_overload()"
fn test_static_overload():
    var a = MyStruct()
    # CHECK-NEXT: %a = lit.varlet.decl
    # CHECK-NEXT: lit.call{{.*}}__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call @{{.*}}foo{{.*}}(%a)
    a.foo()


struct VariadicStruct[*Ts: AnyRegType]:
    fn __init__(inout self):
        pass

    @staticmethod
    fn param_func[i: Int]():
        pass


fn take_variadic_struct[*Ts: AnyRegType](a: VariadicStruct[Ts]):
    pass


# CHECK-LABEL: lit.func @"variadic_params()"
fn variadic_params():
    # CHECK-NEXT: call {{.*}}param_func[{{.*}}$int::Int]()"<:variadic<regtype> [!Int, {{.*}}SIMD{{.*}}f32}>, :!Int {{.*}}1
    VariadicStruct[Int, Float32].param_func[4]()
    # CHECK: call {{.*}}take_variadic_struct{{.*}}<:variadic<regtype> [!Int, {{.*}}SIMD{{.*}}f32
    take_variadic_struct(VariadicStruct[Int, Float32]())


# Test that pointers don't get confused with by-ref arguments.
# CHECK-LABEL: lit.func @"testPointerArgs{{.*}}(%ptr[ptr]: !kgen.pointer<si32> borrow) -> si32
fn testPointerArgs(ptr: __mlir_type.`!kgen.pointer<si32>`) -> __mlir_type.si32:
    # CHECK-NEXT: %0 = pop.load %ptr : !kgen.pointer<si32>
    return __mlir_op.`pop.load`[_type=__mlir_type.si32](ptr)


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
    # CHECK: %0 = lit.call @"$decls"::@"math(
    # CHECK: lit.return %0 : index
    return math(a, math(Int(1).value, Int(2).value))


@always_inline("nodebug")
fn returnParameter[a: __mlir_type.index]() -> __mlir_type.index:
    return a


# CHECK-LABEL: lit.func @"callReturnParam
fn callReturnParam() -> __mlir_type.index:
    # CHECK-NEXT: %0 = lit.call @"$decls"::@"returnParameter[__mlir_type.index]()"<3>()
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


# CHECK-LABEL: lit.func @"adaptiveNestedFns
fn adaptiveNestedFns(a: Int, b: Int):
    # CHECK: lit.func *"nestedFn{{.*}}"{{.*}}isAdaptive
    @adaptive
    @parameter
    fn nestedFn(d: Int) -> Int:
        return a + d

    # CHECK: lit.func *"nestedFn{{.*}}_0"{{.*}}isAdaptive
    @adaptive
    @parameter
    fn nestedFn(d: Int) -> Int:
        return b + d

    # CHECK: kgen.param.fork *"(adaptive)nestedFn{{.*}}": {{.*}} = <[*"nestedFn{{.*}}", *"nestedFn{{.*}}_0"]>
    # CHECK: call_param[{{.*}}: *"(adaptive)nestedFn{{.*}}"]
    let c = nestedFn(2)


# CHECK-LABEL: lit.func @"nestedFnInLoop()"
fn nestedFnInLoop():
    # CHECK: lit.loop
    for i in range(10):
        # CHECK: lit.call @{{.*}}__next__
        # CHECK: lit.func *"foo()"
        @always_inline
        @noncapturing
        fn foo() -> Int:
            # CHECK: %[[I:.*]] = lit.ref.load %i
            # CHECK-NEXT: return %[[I]]
            return i

        # CHECK: lit.call_param[!lit.signature<() -> !Int>: *"foo()"]()
        let result = foo()


fn paramRefFunc[T: AnyRegType](x: T):
    pass


# CHECK-LABEL: lit.func @"orvalueInferType()"
fn orvalueInferType():
    @noncapturing
    fn func(x: __mlir_type.index) -> __mlir_type.index:
        return x

    # CHECK: call {{.*}}paramRefFunc{{.*}}<:regtype !lit.signature<("x": index borrow) -> index>>
    paramRefFunc(func)


# CHECK-LABEL: lit.func @"kernel
# CHECK-SAME: <[[X:.*]][x]: !Int>
# CHECK-SAME: LLVMMetadata = {nvvm.maxntid = {{.*}}#kgen.pack<[[X]]> : !kgen.pack<[!Int]>
@__llvm_metadata(`nvvm.maxntid`=[x])
fn kernel[x: Int]():
    pass

# https://github.com/modularml/mojo/issues/1152
# Allow mutable self argument when overloading operators using dunder methods
struct MutatingAdd:
  fn __add__(inout self, x: MutatingAdd): pass

# CHECK-LABEL: lit.func @"testMutatingAdd
fn testMutatingAdd(owned a: MutatingAdd, b: MutatingAdd):
  # CHECK-NEXT: lit.call {{.*}}__add__{{.*}}(%a, %b)
  a + b


##===----------------------------------------------------------------------===##
# Conventions
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"ownedConventionMem
# CHECK-SAME: (%a[a]: !lit.ref<mut !StructWithInit, {{.*}}> owned_in_mem,
# CHECK-SAME:  %b[b]: !lit.ref<mut !StructWithInit, {{.*}}> borrow_in_mem)
fn ownedConventionMem(owned a: StructWithInit, borrowed b: StructWithInit):
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK: %1 = lit.ref.load [[AX]]
    _ = a.x
    # CHECK: [[BY:%.*]] = lit.ref.struct.ger %b[y]
    # CHECK: = lit.ref.load [[BY]]
    _ = b.y

    # It is ok to mutate owned values.
    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a[x]
    # CHECK-NEXT: [[FOUR:%.*]] = kgen.param.constant: {{.*}} = 4
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
# CHECK-SAME: (%a[a]: !RPStructWithInit,
# CHECK-SAME:  %b[b]: !RPStructWithInit borrow,
# CHECK-SAME:  %triv[triv]: !RPStructWithInitTrivial borrow)
fn ownedConventionReg(
    owned a: RPStructWithInit,
    borrowed b: RPStructWithInit,
    borrowed triv: RPStructWithInitTrivial,
):
    # CHECK: %a_0 = lit.varlet.decl "a" imp
    # CHECK: lit.ref.store %a, %a_0

    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a_0[x]
    # CHECK:  = lit.ref.load [[AX]]
    _ = a.x
    # CHECK: [[BY:%.*]] = lit.struct.extract %b[y]
    _ = b.y

    # CHECK: %t = lit.letreg.decl "t" = %triv
    # No copy call.
    let t = triv

    # CHECK: [[AX:%.*]] = lit.ref.struct.ger %a_0[x]
    # CHECK: [[ONE:%.*]]  = kgen.param.constant: !Int = <#lit.struct<{value = 1}>>
    # CHECK: lit.ref.store [[ONE]], [[AX]]
    a.x = 1


struct BorrowStruct:
    fn testMethod(borrowed self):
        pass

    fn borrowedVarArgs(borrowed self, borrowed *x: BorrowStruct):
        pass


# CHECK-LABEL: callerFn
# CHECK-SAME: (%arg0[arg0]: !lit.ref<{{.*}}> borrow_in_mem)
fn callerFn(borrowed arg0: BorrowStruct):
    # CHECK-NEXT: lit.call {{.*}}testMethod{{.*}}(%arg0)
    arg0.testMethod()

    # CHECK: %1 = pop.variadic.create [%arg0, %arg0]
    # CHECK: lit.call {{.*}}borrowedVarArgs{{.*}}(%arg0, %1)
    arg0.borrowedVarArgs(arg0, arg0)


##===----------------------------------------------------------------------===##
# Default arguments and variadics.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"defaultArgument
# CHECK-SAME: %c[c]: !Int borrow = #lit.struct<{value = 5}>)
fn defaultArgument(a: Int, b: Int = 3, c: Int = 5) -> Int:
    return a + b


# CHECK-LABEL: lit.func @"callDefaultArgument
fn callDefaultArgument(x: Int) -> Int:
    # CHECK-NEXT: %[[ARG1:.*]] = kgen.param.constant{{.*}} = 3
    # CHECK-NEXT: %[[ARG2:.*]] = kgen.param.constant{{.*}} = 5
    # CHECK-NEXT: lit.call {{.*}}defaultArgument{{.*}}(%x, %[[ARG1]], %[[ARG2]])
    # CHECK-NEXT: lit.letreg.decl "a"
    let a = defaultArgument(x)
    # CHECK-NEXT: %[[ARG2:.*]] = kgen.param.constant{{.*}} = 5
    # CHECK-NEXT: lit.call {{.*}}defaultArgument{{.*}}(%x, %x, %[[ARG2]])
    let b = defaultArgument(x, x)
    return a + b


# CHECK-LABEL: lit.func @"defaultArgumentReferencesParameter
# CHECK-SAME: (%a[a]: !Int borrow = apply(:!lit.signature<("self": !Int borrow, "rhs": !Int borrow)
# CHECK-SAME: -> !Int> {{.*}}Int::@"__add__({{.*}}$int::Int,{{.*}}$int::Int)", {{.*}}p, #lit.struct<{value = 87}>))
fn defaultArgumentReferencesParameter[p: Int](a: Int = p + 87) -> Int:
    return a

# CHECK-LABEL: lit.func @"defaultArgumentUntyped
# CHECK-SAME: owned_in_mem = apply_result_slot({{.*}}object::@"__init__
def defaultArgumentUntyped(a = 1): pass

struct MemoryType:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

# CHECK-LABEL: lit.func @"defaultArgumentNonRegisterType
# CHECK-SAME: borrow_in_mem = apply_result_slot({{.*}}__init__
fn defaultArgumentNonRegisterType(a: MemoryType = 1): pass

# CHECK-LABEL: lit.func @"callNonRegisterDefaultArg
fn callNonRegisterDefaultArg():
    # CHECK: %[[ANON:.*]] = lit.varlet.decl "anonymous*" synth : !lit.ref<mut !MemoryType, *"`anonymous*0">
    # CHECK: %[[VALUE:.*]] = kgen.param.materialize: !MemoryType = <apply_result_slot({{.*}}value = 1
    # CHECK: lit.ref.store %[[VALUE]], %[[ANON]]
    # CHECK: call {{.*}}defaultArgumentNonRegisterType{{.*}}(%anonymous2A)
    defaultArgumentNonRegisterType()
    # CHECK: lit.alias.decl {{.*}}none: none = <apply({{.*}}defaultArgumentNonRegisterType
    # CHECK-SAME: store_to_mem(apply_result_slot({{.*}}MemoryType::@"__init__{{.*}}value = 1}>
    alias none = defaultArgumentNonRegisterType()

# CHECK: lit.func @"referencesDefaultArgumentFunction
fn referencesDefaultArgumentFunction():
    # CHECK: %f = lit.letreg.decl "f" = %0
    let f = defaultArgument


# CHECK-LABEL: lit.func @"variadics({{.*}}$int::Int*)"(%a[a]: !kgen.variadic<!Int> borrow) vararg
fn variadics(*a: Int):
    # CHECK-NEXT: %[[LIST:.*]] = lit.call {{.*}}VariadicList{{.*}}__init__
    # CHECK-NEXT: lit.letreg.decl "a" {{.*}}%[[LIST]]
    let size = len(a)
    let elt0 = a[0]
    let elt1 = a[1]


fn parameterizedVariadic[T: __mlir_type.`!kgen.anyregtype`](*args: T):
    pass


struct ParameterizedStruct[T: __mlir_type.`!kgen.anyregtype`]:
    fn __init__(inout self, *args: T):
        pass


struct VarArgsParameterizedStruct[*Is: Int]:
    fn __init__(inout self):
        pass


# CHECK-LABEL: lit.func @"callVariadic{{.*}})"<
# CHECK-SAME: [[P:.*_p]][p]: !Int>
fn callVariadic[p: Int](x: Int):
    # CHECK: %variadic = kgen.param.constant: variadic<!Int> = <[]>
    # CHECK: call @"$decls"::@"variadics($builtin::$int::Int*)"(%variadic)
    variadics()
    # CHECK: %variadic_0 = kgen.param.constant: variadic<!Int> = <[{{.*}}7{{.*}}11{{.*}}13{{.*}}]>
    # CHECK: call @"$decls"::@"variadics($builtin::$int::Int*)"(%variadic_0)
    variadics(7, 11, 13)
    # CHECK: %[[VAR:.*]] = pop.variadic.create [%x]
    # CHECK: call @"$decls"::@"variadics($builtin::$int::Int*)"(%[[VAR]])
    variadics(x)
    # CHECK: %[[CST:.*]] = kgen.param.constant: !Int
    # CHECK: %[[VAR:.*]] = pop.variadic.create [%x, %[[CST]]]
    # CHECK: call @"$decls"::@"variadics($builtin::$int::Int*)"(%[[VAR]])
    variadics(x, 1)

    # CHECK: @"variadics($builtin::$int::Int*)", []
    alias EmptyVariadic = variadics()
    # CHECK: @"variadics($builtin::$int::Int*)", [[[P]], {{.*}} = 1{{.*}}]
    alias NonEmptyVariadic = variadics(p, 1)

    # CHECK: @"parameterizedVariadic{{.*}}"<:regtype !Int>
    parameterizedVariadic(1, 2)
    # CHECK: lit.call {{.*}}@ParameterizedStruct::@"__init__(${{.*}}::ParameterizedStruct[[[T:.*]]]=&,[[T]]*)"{{.*}}<:regtype !Int>
    _ = ParameterizedStruct(3)
    # CHECK: lit.call {{.*}}@VarArgsParameterizedStruct::@"__init__(${{.*}}::VarArgsParameterizedStruct[[[IS:.*]]]=&)"{{.*}}<:variadic<!Int> [#lit.struct<{value = 4}>, #lit.struct<{value = 5}>]>
    _ = VarArgsParameterizedStruct[4, 5]()
    # CHECK: lit.call {{.*}}@VarArgsParameterizedStruct::@"__init__(${{.*}}::VarArgsParameterizedStruct[[[IS]]]=&)"{{.*}}<:variadic<!Int> []>
    _ = VarArgsParameterizedStruct()


# CHECK-LABEL: lit.struct.decl @MyTuple
# CHECK-SAME: <[[TUPLETS:.*]][Ts]: variadic<regtype>>
struct MyTuple[*Ts: __mlir_type.`!kgen.anyregtype`]:
    var elements: __mlir_type[`!kgen.pack<`, Ts, `>`]

    fn __init__(inout self, *args: *Ts):
        self.elements = args


# CHECK-LABEL: lit.func @"pack[__mlir_type.!kgen.variadic<regtype>](__mlir_type.!kgen.pack<*(0,0)>)"<
# CHECK-SAME: [[TS:.*_Ts]][Ts]: variadic<regtype>>(%args[args]: !kgen.pack<[[TS]]>)
fn pack[*Ts: __mlir_type.`!kgen.anyregtype`](owned *args: *Ts):
    # CHECK: %copy = lit.letreg.decl "copy" = %args : !kgen.pack<[[TS]]>
    let copy = args


# CHECK-LABEL: lit.func @"packBorrowed{{.*}})"<
# CHECK-SAME: [[TS:.*_Ts]][Ts]: variadic<regtype>>
fn packBorrowed[*Ts: __mlir_type.`!kgen.anyregtype`](borrowed *args: *Ts):
    # CHECK: %copy = lit.letreg.decl "copy" = %args : !kgen.pack<[[TS]]>
    let copy = args


# Ensure that parameters can be bound correctly.
fn variadicParameter[*Ts: __mlir_type.`!kgen.anyregtype`](x: Int):
    pass


# CHECK-LABEL: lit.func @"usePacks
# CHECK-SAME: [[ARGX:%.*]][x]: !kgen.declref<@"$builtin"::@"$simd"::@SIMD{{.*}}f32
# CHECK-SAME: [[ARGY:%.*]][y]: !Int
fn usePacks(x: Float32, y: Int):
    # CHECK: lit.varlet.decl {{.*}} : !lit.ref<mut @"$decls"::@MyTuple<:variadic<regtype> [!Int]>
    var a: MyTuple[Int]
    # CHECK: lit.varlet.decl {{.*}} : !lit.ref<mut @"$decls"::@MyTuple<:variadic<regtype> [!Int, @"$builtin"::@"$simd"::@SIMD{{.*}}f32{{.*}}, !Int]>
    var b: MyTuple[Int, Float32, Int]
    # CHECK: lit.varlet.decl {{.*}} : !lit.ref<mut @"$decls"::@MyTuple<:variadic<regtype> [!Int]>
    let c = MyTuple[Int](1)
    # CHECK: lit.varlet.decl {{.*}} : !lit.ref<mut @"$decls"::@MyTuple<:variadic<regtype> [!FloatLiteral, index]>
    let d = MyTuple(3.14, Int(6).value)
    # CHECK: lit.varlet.decl {{.*}} : !lit.ref<mut @"$decls"::@MyTuple<:variadic<regtype> []>
    let e = MyTuple()

    # CHECK: %[[PACK1:.*]] = kgen.param.constant: !kgen.pack<[index]> = <<1>>
    # CHECK: lit.call @"$decls"::@"pack{{.*}}(%[[PACK1]])
    pack(Int(1).value)
    # CHECK: %[[PACK2:.*]] = kgen.param.constant: !kgen.pack<[index, {{.*}}FloatLiteral, index]> = <<1, {{.*}}3.14{{.*}}, 2>>
    # CHECK: lit.call @"$decls"::@"pack{{.*}} [index, {{.*}}FloatLiteral, index]>(%[[PACK2]])
    pack(Int(1).value, 3.14, Int(2).value)
    # CHECK: %[[PACK3:.*]] = kgen.param.constant: !kgen.pack<[]> = <<>>
    # CHECK: lit.call @"$decls"::@"pack{{.*}} []>(%[[PACK3]])
    pack()

    # CHECK: %[[PACK4:.*]] = kgen.pack.create(%{{.*}}, [[ARGX]], [[ARGY]])
    # CHECK: lit.call @"$decls"::@"pack{{.*}} [index, @"$builtin"::@"$simd"::@SIMD{{.*}}f32{{.*}}, !Int]>(%[[PACK4]])
    pack(Int(1).value, x, y)
    # CHECK: %[[INTCTOR:.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 1}>>
    # CHECK: %[[PACK5:.*]] = kgen.pack.create(%[[INTCTOR]], %x, %y)
    # CHECK: lit.call @"$decls"::@"pack{{.*}} [!Int, @"$builtin"::@"$simd"::@SIMD{{.*}}f32{{.*}}, !Int]>(%[[PACK5]])
    pack[Int, Float32, Int](Int(1).value, x, y)

    # CHECK: kgen.param.constant = <1>
    # CHECK-NEXT: [[PACK6:%.*]] = kgen.pack.create(%{{.*}}, [[ARGX]], [[ARGY]])
    # CHECK-NEXT: lit.call {{.*}}packBorrowed{{.*}}([[PACK6]])
    packBorrowed(Int(1).value, x, y)

    # CHECK: lit.call {{.*}}variadicParameter{{.*}}<:variadic<regtype>  [!Int, @"$builtin"::@"$simd"::@SIMD{{.*}}f32{{.*}}]>
    variadicParameter[Int, Float32](1)
    # CHECK: lit.call {{.*}}variadicParameter{{.*}}<:variadic<regtype> []>
    variadicParameter(Int(2).value)


# COM: Test variadic arguments in a parameter context.
@value
struct MemStruct:
    alias t = 5

fn variadic_mem_only(*values: MemStruct) -> Int:
    return __get_value_from_ref(values[1]).t

# CHECK-LABEL: lit.func @"test_variadic_mem_only{{.*}}"<
# CHECK-SAME: [[X:.*]][x]: !MemStruct, [[Y:.*]][y]: !MemStruct>()
fn test_variadic_mem_only[x: MemStruct, y: MemStruct]():
    # CHECK: lit.alias.decl {{.*}}: !Int = <apply(
    # CHECK-SAME: :(!kgen.variadic<!lit.ref<mut !MemStruct, #lit.lifetime>> borrow_in_mem) vararg -> !Int {{.*}}::@"variadic_mem_only({{.*}}::MemStruct*)"
    # CHECK-SAME: [store_to_mem([[X]]), store_to_mem([[Y]])]
    alias b = variadic_mem_only(x, y)


# CHECK-LABEL: lit.func @"implicit_return_obj
# CHECK-SAME: object{{.*}} byref_result
def implicit_return_obj():
    # CHECK: if
    if False:
        # CHECK: lit.call {{.*}}object::@"__init__{{.*}}%__result__
        # CHECK: kgen.variant.create
        # CHECK: return
        return
    # CHECK: else
    else:
        # CHECK: lit.call {{.*}}object::@"__init__{{.*}}%__result__
        # CHECK: kgen.variant.create
        # CHECK: return
        return 5
    # CHECK: lit.call {{.*}}object::@"__init__
    # CHECK: kgen.variant.create
    # CHECK: lit.return
    _ = 5


##===----------------------------------------------------------------------===##
# raises specifier.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"defAlwaysRaises()"() throws -> !kgen.variant<!Error, !Int> attributes {isDef
def defAlwaysRaises() -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}#lit.struct<{value = 0}>
    # CHECK-NEXT: %1 = kgen.variant.create [[RESULT]]
    # CHECK-NEXT: lit.return %1
    return 0


# CHECK-LABEL: lit.func @"fnThatRaises()"() throws -> !kgen.variant<!Error, !Int>
fn fnThatRaises() raises -> Int:
    # CHECK: [[RESULT:%.*]] = kgen{{.*}}#lit.struct<{value = 0}>
    # CHECK-NEXT: %1 = kgen.variant.create [[RESULT]]
    # CHECK-NEXT: lit.return %1
    return 0


# CHECK-LABEL: lit.func @"raisesReturnsNone()"() throws -> !kgen.variant<!Error, none>
fn raisesReturnsNone() raises:
    # CHECK-NEXT: %none = kgen.param.constant: none
    # CHECK-NEXT: %0 = kgen.variant.create %none
    # CHECK-NEXT: lit.return %0
    # CHECK-NEXT: lit.end_func
    pass


# COM: We can return an variant of error and index in a non-throwing function.
# CHECK-LABEL: lit.func @"raisesReturnsVariant()"() -> !kgen.variant<!Error, index>
fn raisesReturnsVariant() -> __mlir_type[`!kgen.variant<`, Error, `, index>`]:
    return __mlir_op.`kgen.variant.create`[
        _type=__mlir_type[`!kgen.variant<`, Error, `, index>`],
        index=Int(1).value
    ](Int(1).value)


# CHECK-LABEL: lit.func @"raise_and_return
# CHECK-SAME: -> !kgen.variant<!Error, !Error>
fn raise_and_return(a: Error) raises -> Error:
  # COM: Index 1 is the success index.
  # CHECK: kgen.variant.create %{{.*}}, 1 : <!Error, !Error>
  return a


@value
@register_passable("trivial")
struct RaisingGetterSetter:
    fn __getitem__(self, i: Int) raises -> Float32:
        return 1

    fn __setitem__(inout self, i: Int, v: Float32) raises:
        pass


fn test_raising_computed_getter() raises:
    let a = RaisingGetterSetter()[2]


##===----------------------------------------------------------------------===##
# Structs
##===----------------------------------------------------------------------===##

fn forward_ref(x: EmptyStruct):
  pass


# CHECK-LABEL: lit.struct.decl @EmptyStruct register_passable {
@register_passable
struct EmptyStruct:
    pass


# CHECK-NEXT: }

# CHECK-LABEL: lit.struct.decl @OneLineStruct<{{.*}}[size]: !Int> {
struct OneLineStruct[size: Int]:
    pass
    pass


# CHECK-NEXT: }

# CHECK-LABEL: lit.struct.decl @StructWithInit
struct StructWithInit:
    var x: Int
    var y: Int

    # CHECK: lit.func @"__init__($decls::StructWithInit=&,{{.*}}$int::Int)"
    # CHECK-SAME: (%self[self]: !lit.ref<mut !StructWithInit, {{.*}}> init_self,
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
    # CHECK-SAME: (%self[self]: !lit.ref<mut !StructWithInit, {{.*}}> init_self,
    fn __init__(inout self, a: Int, b: Int):
        # CHECK: hlcf.if
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

    fn __init__() -> Self:
        return Self {}

    # CHECK: lit.func @"static({{.*}}$int::Int)"(%x[x]: !Int borrow) -> !kgen.none attributes {{.*}} isStatic
    @staticmethod
    fn static(x: Int):
        # CHECK: %0 = {{.*}}#lit.struct<{value = 4}>
        # CHECK: lit.call @"$decls"::@StructExample::@"static{{.*}}"(%0)
        StructExample.static(4)
        pass

    # CHECK: lit.func @"mutatingMethod{{.*}}(%self[self]: !lit.ref<mut !StructExample, {{.*}}> byref) -> !kgen.none
    fn mutatingMethod(inout self):
        pass


# CHECK: lit.func @"callStatic{{.*}}(%a[a]: !Int borrow)
fn callStatic(a: Int):
    # CHECK: lit.call @"$decls"::@StructExample::@"static{{.*}}(%a)
    StructExample.static(a)

    # CHECK: lit.call @"$decls"::@StructExample::@"__init__{{.*}}()
    # CHECK: lit.call @"$decls"::@StructExample::@"static{{.*}}(%a)
    StructExample().static(a)


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

# CHECK-LABEL: lit.struct.decl @ValueMem(trait<@{{.*}}::@Copyable>, trait<@{{.*}}::@Movable>) attributes {
# CHECK-SAME: moveInit = #kgen.symbol.constant<{{.*}}ValueMem::@"__moveinit__
# CHECK-SAME: !kgen.signature<!lit.signature<[2]({{.*}} init_self, {{.*}} owned_in_mem, |)
@value
struct ValueMem:
    var a: Int  # Trivial
    var b: StructExample  # Copy ctor


# CHECK: lit.func @"__init__(
# CHECK-SAME:  %[[SELF:.*]][*""]: !lit.ref<mut !ValueMem, {{.*}}> init_self,
# CHECK-SAME:  %a[a]: !Int borrow,
# CHECK-SAME:  %b[b]: !StructExample
# CHECK-SAME: ) -> !kgen.none attributes {isSynthetic, sourceName = "__init__", specialFnKind = 2 : i8} {
# CHECK-NEXT: %[[PA:.*]] = lit.ref.struct.ger %[[SELF]][a]
# CHECK-NEXT: lit.ref.store %a, %[[PA]]
# CHECK-NEXT: %[[PB:.*]] = lit.ref.struct.ger %[[SELF]][b]
# CHECK-NEXT: lit.ref.store %b, %[[PB]]
# CHECK-NEXT: kgen.param.constant: none

# CHECK: lit.func @"__copyinit__(
# CHECK-SAME:  %self[self]: !lit.ref<mut !ValueMem, {{.*}}> init_self,
# CHECK-SAME:  %other[other]: !lit.ref<!ValueMem, {{.*}}> borrow_in_mem, |)
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %other[a]
# CHECK-NEXT: %2 = lit.ref.load %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %other[b]
# CHECK-NEXT: %5 = lit.ref.load %4
# CHECK-NEXT: %6 = lit.call {{.*}}__copyinit__{{.*}}(%5)
# CHECK-NEXT: lit.ref.store %6, %3
# CHECK-NEXT: kgen.param.constant: none

# CHECK: lit.func @"__moveinit__(
# CHECK-SAME:  %self[self]: !lit.ref<mut !ValueMem, {{.*}}> init_self,
# CHECK-SAME:  %other[other]: !lit.ref<mut !ValueMem, {{.*}}> owned_in_mem, |)
# CHECK-NEXT: %0 = lit.ref.struct.ger %self[a]
# CHECK-NEXT: %1 = lit.ref.struct.ger %other[a]
# CHECK-NEXT: %2 = lit.load.consume %1
# CHECK-NEXT: lit.ref.store %2, %0
# CHECK-NEXT: %3 = lit.ref.struct.ger %self[b]
# CHECK-NEXT: %4 = lit.ref.struct.ger %other[b]
# CHECK-NEXT: %5 = lit.load.consume %4
# CHECK-NEXT: lit.ref.store %5, %3
# CHECK-NEXT: kgen.param.constant: none

# CHECK-LABEL: lit.struct.decl @ValueMemHasCopy(trait<@{{.*}}::@Copyable>, trait<@{{.*}}::@Movable>) attributes {
@value
struct ValueMemHasCopy:
    var a: Int
    var b: StructExample
    fn __copyinit__(inout self, other: Self):
       self.a = other.a
       self.b = other.b

# CHECK-LABEL: lit.struct.decl @ValueMemHasMove(trait<@{{.*}}::@Copyable>, trait<@{{.*}}::@Movable>) attributes {
@value
struct ValueMemHasMove:
    var a: Int
    var b: StructExample
    fn __moveinit__(inout self, owned other: Self):
       self.a = other.a
       self.b = other.b

# CHECK-LABEL: lit.struct.decl @ValueRegTrivial register_passable_trivial {
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


# CHECK: lit.func @"__init__(
# CHECK-SAME:  %a[a]: !Int borrow,
# CHECK-SAME:  %b[b]: !StructExample
# CHECK-SAME: ) ownedresult -> !ValueReg
# CHECK-NEXT: %0 = lit.struct.create(a=%a, b=%b)
# CHECK-NEXT: lit.return %0
# CHECK-NEXT: lit.end_func

# CHECK: lit.func @"__copyinit__
# CHECK-SAME: (%other[other]: !ValueReg borrow, |)
# CHECK-SAME:  -> !ValueReg
# CHECK-SAME: attributes {{.*}}specialFnKind = 6 : i8
# CHECK-NEXT: %0 = lit.struct.extract %other[a]
# CHECK-NEXT: %1 = lit.struct.extract %other[b]
# CHECK-NEXT: %2 = lit.call {{.*}}__copyinit__{{.*}}(%1)
# CHECK-NEXT: %3 = lit.struct.create(a=%0, b=%2)
# CHECK-NEXT: lit.return %3

# COM: Ensure that "self" is a valid field name.
# CHECK-LABEL: lit.struct.decl @Foo(trait<@{{.*}}::@Copyable>, trait<@{{.*}}::@Movable>) attributes
@value
struct Foo:
    var a: Int
    var self: Int


# CHECK: lit.func @"__init__{{.*}}(%[[SELFARG:.*]][*""]: !lit.ref<mut !Foo, {{.*}}> init_self, |, %a[a]: !Int borrow, %self[self]: !Int borrow)

# CHECK-LABEL: lit.struct.decl @ParamVarArg
@value
@register_passable("trivial")
struct ParamVarArg[*I: Int]:
    # CHECK: lit.func @"__init__
    # CHECK-SAME: param_vararg
    pass

# CHECK-LABEL: lit.struct.decl @TraitMember
@value
struct TraitMember[T: Copyable]:
    var value: T
    # CHECK: lit.func @"__copyinit__
    # CHECK: call_param{{.*}}__copyinit__
    # CHECK: lit.func @"__moveinit__
    # CHECK: call_param{{.*}}__copyinit__

# CHECK: lit.func @"notSynthetic{{.*}}(%self[self]: !lit.ref<mut !NotSynthetic, {{.*}}> borrow_in_mem) -> !kgen.none attributes {isParametric, sourceName = "notSynthetic", specialFnKind = 0 : i8}
# CHECK: lit.func @"__init__{{.*}}isSynthetic
# CHECK: lit.func @"__copyinit__{{.*}}isSynthetic
# CHECK: lit.func @"__moveinit__{{.*}}isSynthetic
@value
struct NotSynthetic:
   var member: __mlir_type.`index`
   fn notSynthetic(self):
      pass

##===----------------------------------------------------------------------===##
# async/await
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"coroutine()"() async -> !Int
async fn coroutine() -> Int:
    # CHECK: lit.return %0
    return 0


# CHECK-LABEL: lit.struct.decl @StructWithAsync
struct StructWithAsync:
    # CHECK-LABEL: lit.func @"do_something{{.*}}({{.*}}) async
    async fn do_something(self: StructWithAsync):
        # CHECK-NEXT: %[[CORO:.*]] = lit.async.call[!lit.signature<() async -> !Int>: @"$decls"::@"coroutine()"]()
        # CHECK-NEXT: %[[COROUTINE:.*]] = lit.call {{.*}}@Coroutine::@"__init__{{.*}}<:regtype !Int>(%[[CORO]])
        # CHECK-NEXT: lit.letreg.decl "a" = %[[COROUTINE]]
        let a = coroutine()


# CHECK-LABEL: lit.func @"throwing_coroutine()"() throws|async -> !kgen.variant<!Error, !Int>
async fn throwing_coroutine() raises -> Int:
    raise Error("oh no!")


# CHECK-LABEL: lit.func @"call_raising_coro()"
fn call_raising_coro():
    # CHECK: %[[CORO:.*]] = lit.async.call[{{.*}}throwing_coroutine
    # CHECK-NEXT: call {{.*}}RaisingCoroutine::@"__init__{{.*}}<:regtype !Int>(%[[CORO]])
    let coro = throwing_coroutine()


# CHECK-LABEL: lit.func @"call_struct_async{{.*}}({{.*}}) async -> !kgen.none
async fn call_struct_async(f: StructWithAsync):
    # CHECK-NEXT: lit.async.call[!lit.signature<[1]({{.*}}) async -> !kgen.none>: @{{.*}}](%f)
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

    # CHECK: lit.alias.decl {{.*}}b: !lit.signature<() capturing -> !Int> = <*"nestedFunction()">
    alias b = nestedFunction
    # CHECK: call_param[!lit.signature<() capturing -> !Int>: *"nestedFunction()"]()
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

        # CHECK: lit.alias.decl {{.*}}b: !lit.signature<() capturing -> !Int> = <*"nestedFunction()">
        alias b = nestedFunction
        # CHECK: call_param[!lit.signature<() capturing -> !Int>: *"nestedFunction()"]()
        return nestedFunction()


# CHECK-LABEL: lit.func @"closureParameter[fn() capturing -> __mlir_type.index]()"
# CHECK-SAME: capturing ->
fn closureParameter[
    func: fn() capturing -> __mlir_type.index
]():
    pass


# CHECK-LABEL: lit.func @"topLevelParamFn[__mlir_type.index]()"<{{.*}}[a_param]>
fn topLevelParamFn[a_param: __mlir_type.index]():
    # CHECK: lit.func *"nestedFunction[__mlir_type.index]()"<{{.*}}[b_param]>
    @noncapturing
    fn nestedFunction[b_param: __mlir_type.index]():
        return

    # CHECK: lit.alias.decl {{.*}}thinref: !lit.signature<<"b_param": index>() -> !kgen.none> = <*"nestedFunction[__mlir_type.index]()">
    alias thinref = nestedFunction
    # CHECK: call_param[{{.*}}: bind_signature(:!lit.signature<<"b_param": index>() -> !kgen.none> *"nestedFunction[__mlir_type.index]()", 2)]()
    nestedFunction[Int(2).value]()

    let value = 0

    @parameter
    fn capturingNestedFunction() -> Int:
        return value

    # CHECK: lit.alias.decl {{.*}}fatRef: !lit.signature<() capturing -> !Int> = <*"capturingNestedFunction()">
    alias fatRef = capturingNestedFunction


struct SomeParamStruct[c_param: Int]:
    # CHECK-LABEL: lit.func @"topLevelParamFn[{{.*}}$int::Int]{{.*}}<{{.*}}[a_param]
    fn topLevelParamFn[a_param: Int](self):
        # CHECK: lit.func *"nestedFunction[{{.*}}$int::Int]()"<{{.*}}[b_param]
        @noncapturing
        fn nestedFunction[b_param: Int]():
            return

        # CHECK: lit.alias.decl {{.*}}reff: !lit.signature<<"b_param": !Int>() -> !kgen.none> = <*"nestedFunction[{{.*}}$int::Int]()">
        alias reff = nestedFunction
        # CHECK: call_param[{{.*}}: bind_signature(:!lit.signature<<"b_param": !Int>() -> !kgen.none> *"nestedFunction[{{.*}}$int::Int]()", {{.*}}2{{.*}})]()
        nestedFunction[2]()


##===----------------------------------------------------------------------===##
# Tuple Types
##===----------------------------------------------------------------------===##

# FIXME: Empty tuple `Tuple[]` cannot be spelled.

# CHECK-LABEL: lit.func @"returnTup0
# CHECK-SAME: -> !kgen.declref<{{.*}}@"$tuple"::@Tuple<:variadic<regtype> []>
fn returnTup0() -> Tuple:
  # FIXME: Why isn't this a kgen.param.constant for the whole call?
  # CHECK-NEXT: %0 = kgen.param.constant: !kgen.pack<[]> = <<>>
  # CHECK-NEXT: %1 = lit.call{{.*}}__init__
  # CHECK-NEXT: lit.return
  return ()

# CHECK-LABEL: lit.func @"returnTup0a
# CHECK-SAME: -> !kgen.declref<{{.*}}@"$tuple"::@Tuple<:variadic<regtype> []>
fn returnTup0a() -> ():
  # FIXME: Why isn't this a kgen.param.constant for the whole call?
  # CHECK-NEXT: %0 = kgen.param.constant: !kgen.pack<[]> = <<>>
  # CHECK-NEXT: %1 = lit.call{{.*}}__init__
  # CHECK-NEXT: lit.return
  return ()

# CHECK-LABEL: lit.func @"returnTup1
# CHECK-SAME: -> !kgen.declref<{{.*}}@"$tuple"::@Tuple<:variadic<regtype> [!Int]>
fn returnTup1() -> Tuple[Int]:
  # CHECK-NEXT: %0 = kgen.param.constant: !kgen.pack<[!Int]>
  # CHECK-NEXT: %1 = lit.call{{.*}}__init__
  # CHECK-NEXT: lit.return
  return (Int(4),)

# CHECK-LABEL: lit.func @"returnTup1
# CHECK-SAME: -> !kgen.declref<{{.*}}@"$tuple"::@Tuple<:variadic<regtype> [!Int]>
fn returnTup1a() -> (Int,):
  return (Int(4),)

fn returnTup1b() -> (Int,):
  return Int(4),

# CHECK-LABEL: lit.func @"returnTup2
# CHECK-SAME: -> !kgen.declref<{{.*}}@"$tuple"::@Tuple<{{.*}}:variadic<regtype> [!Int, !FloatLiteral]>
fn returnTup2() -> Tuple[Int, FloatLiteral]:
  # CHECK-NEXT: kgen.param.constant: !kgen.pack<[!Int, !FloatLiteral]> = <<#lit.struct<{value = 4}>, #lit.struct<{value: scalar<f64> = "2"}>>>
  return (Int(4), 2.0)

# CHECK-LABEL: lit.func @"returnTup2a
# CHECK-SAME: -> !kgen.declref<{{.*}}@"$tuple"::@Tuple<{{.*}}:variadic<regtype> [!Int, !FloatLiteral]>
fn returnTup2a() -> (Int, FloatLiteral):
  # CHECK-NEXT: kgen.param.constant: !kgen.pack<[!Int, !FloatLiteral]> = <<#lit.struct<{value = 4}>, #lit.struct<{value: scalar<f64> = "2"}>>>
  return (Int(4), 2.0)

# CHECK-LABEL: lit.func @"returnTup2b
fn returnTup2b() -> (Int, FloatLiteral):
  return Int(4), 2.0

##===----------------------------------------------------------------------===##
# Global Variables
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.globalvar.decl @trivial_global : !Int
# CHECK-NEXT: %0 = lit.globalvar.ref @{{.*}}::@trivial_global : <mut !Int, #lit.lifetime>
# CHECK-NEXT: %1 = kgen.param.constant
# CHECK-NEXT: lit.ref.store %1, %0
var trivial_global: Int = 1
# CHECK-LABEL: lit.globalvar.decl @trivial_global_implicit : !Int
# CHECK-NEXT: %0 = lit.globalvar.ref
# CHECK-NEXT: %1 = kgen.param.constant
# CHECK-NEXT: lit.ref.store %1, %0
var trivial_global_implicit = 1

@value
@register_passable
struct RegType: pass

# CHECK-LABEL: lit.globalvar.decl @reg_global : !RegType
# CHECK-NEXT: %0 = lit.call {{.*}}@RegType::@"__init__()"
# CHECK-NEXT: %1 = lit.globalvar.ref @{{.*}}::@reg_global
# CHECK-NEXT: lit.ref.store %0, %1
let reg_global: RegType = RegType()
# CHECK-LABEL: lit.globalvar.decl @reg_global_implicit : !RegType isVar
# CHECK-NEXT: %0 = lit.call {{.*}}@RegType::@"__init__()"
# CHECK-NEXT: %1 = lit.globalvar.ref
# CHECK-NEXT: lit.ref.store %0, %1
var reg_global_implicit = RegType()

@value
struct MemType: pass

# CHECK-LABEL: lit.globalvar.decl @mem_global {{.*}}
# CHECK-NEXT: [[GLOBAL:%.*]] = lit.globalvar.ref
# CHECK-NEXT:  = lit.call {{.*}}__init__{{.*}}([[GLOBAL]])
let mem_global: MemType = MemType()
# CHECK-LABEL: lit.globalvar.decl @mem_global_implicit {{.*}} isVar
# CHECK-NEXT: %0 = lit.globalvar.ref
# CHECK-NEXT:  = lit.call {{.*}}__init__{{.*}}(%0)
var mem_global_implicit = MemType()

@value
@register_passable
struct DtorRegType:
    fn __del__(owned self): pass

# CHECK-LABEL: lit.globalvar.decl @reg_dtor
# CHECK: }, {
# CHECK-NEXT: %0 = lit.globalvar.ref {{.*}}@reg_dtor
# CHECK-NEXT: %1 = lit.load.consume %0
# CHECK-NEXT:  = lit.call {{.*}}__del__{{.*}}(%1)
var reg_dtor = DtorRegType()

@value
struct DtorMemType:
    fn __del__(owned self): pass

# CHECK-label: lit.globalvar.decl @mem_dtor : !kgen.declref<@"$decls"::@DtorMemType> {
# CHECK: }, {
# CHECK-NEXT: %0 = lit.globalvar.ref
# CHECK-NEXT: lit.call {{.*}}__del__{{.*}}(%0)
var mem_dtor = DtorMemType()

fn borrowGlobalInt(x: Int): pass
fn borrowGlobalReg(x: RegType): pass
fn mutGlobalReg(inout x: RegType): pass
fn copyGlobalMem(owned x: MemType): pass

fn refGlobals():
    # CHECK: [[TRIVIAL:%.*]] = lit.globalvar.ref {{.*}}@trivial_global
    # CHECK-NEXT: [[VALUE:%.*]] = lit.ref.load [[TRIVIAL]]
    # CHECK-NEXT: call {{.*}}borrowGlobalInt{{.*}}([[VALUE]])
    borrowGlobalInt(trivial_global)
    # CHECK: [[REG:%.*]] = lit.globalvar.ref {{.*}}@reg_global
    # CHECK-NEXT: [[VALUE:%.*]] = lit.ref.load [[REG]]
    # CHECK-NEXT: call {{.*}}borrowGlobalReg{{.*}}([[VALUE]])
    borrowGlobalReg(reg_global)
    # CHECK: [[REG_REF:%.*]] = lit.globalvar.ref {{.*}}@reg_global
    # CHECK-NEXT: call {{.*}}mutGlobalReg{{.*}}([[REG_REF]])
    mutGlobalReg(reg_global_implicit)
    # CHECK: [[MEM_REF:%.*]] = lit.globalvar.ref {{.*}}@mem_global
    # CHECK-NEXT: %anonymous2A = lit.varlet.decl {{.*}} : !lit.ref<mut !MemType
    # CHECK-NEXT: [[MEM_REF_IMM:%.*]] = kgen.rebind [[MEM_REF]]
    # CHECK-NEXT: call {{.*}}__copyinit__{{.*}}(%anonymous2A, [[MEM_REF_IMM]])
    # CHECK-NEXT: call {{.*}}copyGlobalMem{{.*}}(%anonymous2A)
    copyGlobalMem(mem_global)

# CHECK: lit.globalvar.decl export @exported_alias {{.*}} {linkageName = "exported_global"}
@export("exported_global")
var exported_alias = 1
# CHECK: lit.globalvar.decl export C @exported_global_var {{.*}} {linkageName = "exported_global_var"}
@export(ABI="C")
var exported_global_var = 1

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
fn not_c_exported(): pass

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
# CHECK-NEXT: decorators <:none apply({{.*}}decorator_arg{{.*}}, #lit.struct<{value = 2}>
@decorator_arg(2)
struct DecoratedStruct:
    pass

##===----------------------------------------------------------------------===##
# Implicit lifetimes for result slots.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"getThing()"
# CHECK-SAME: [*"`__result__"](%__result__[__result__]:
fn getThing() -> MyStruct:
   # result slot parameter should get a different name to avoid conflict.
   # CHECK: lit.func *"localTest()"
   # CHECK-SAME: [*"`__result__0"](%__result___0[__result__]:
   fn localTest() -> MyStruct:
     return MyStruct()
   return localTest()

# CHECK-LABEL: lit.func @"callThing()"
# CHECK-SAME: [*"`__result__"](%__result__[__result__]:
fn callThing() -> MyStruct:
  return getThing()
