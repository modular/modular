# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics | kgen-opt -verify-parameters | FileCheck %s

alias index = __mlir_type.index
alias index_one = __mlir_attr.`1 : index`
alias index_two = __mlir_attr.`2 : index`
alias index_three = __mlir_attr.`3 : index`

##===----------------------------------------------------------------------===##
# Input parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.struct.decl @OurSIMD
# CHECK-SAMEL <[[SIMDSIZE:.*]]: !Int, [[SIMDDT:.*]]: !DType>
# CHECK-SAME: register_passable
@register_passable
struct OurSIMD[size: Int, dt: DType]:
  fn __copyinit__(self) -> Self: pass


@register_passable
struct StructWithIntParam[size: Int]:
  pass

# CHECK-LABEL: lit.func @"paramArith{{.*}}"<{{.*}}[x]: !Int>() -> !kgen.none
fn paramArith[x: Int]():
  # CHECK: lit.alias.decl {{.*}} = <{{.*}}apply({{.*}}__eq__{{.*}}, {{.*}}x, {{.*}}-99{{.*}})>
  alias y = x == -100 + 1

fn take_3index(a: Int, b: Int, c: Int) -> Int:
  return a

# CHECK-LABEL: lit.func @"fancy_signature
# CHECK-SAME: <[[DT:.*_dt]][dt]: !DType, [[SIZE:.*_size]][size]: !Int>
# CHECK-SAME: (%x: {{.*}}@OurSIMD<:!Int [[SIZE]], :!DType [[DT]]>{{.*}}> borrow,
# CHECK-SAME: %exp: {{.*}}@OurSIMD<:!Int [[SIZE]], :!DType [[DT]]>{{.*}}> borrow) -> !Int
fn fancy_signature[dt: DType, size: Int]
  (x: OurSIMD[size, dt], exp: (OurSIMD)[size, dt]) -> Int:

  # CHECK: %local = lit.varlet.decl "local" var
  # CHECK: %[[TMP1:.*]] = kgen.param.constant: !Int = <[[SIZE]]>
  # CHECK: %[[TMP2:.*]] = kgen.param.constant: !Int = <[[SIZE]]>
  # CHECK: %[[TMP3:.*]] = kgen.param.constant: !Int = <[[SIZE]]>
  # CHECK: %[[RES:.*]] = lit.call @"$parameters"::@"take_3index{{.*}}(%[[TMP1]], %[[TMP2]], %[[TMP3]])
  # CHECK: lit.ref.store %[[RES]], %local
  var local = take_3index(size, size, size)

  # CHECK: %[[TMP:.*]] = lit.call {{.*}}__add__
  # CHECK: lit.return %[[TMP]]
  return size+42


fn generic_fn[a: DType, b: Int, c: __mlir_type.`!kgen.type`](d : Int):
  pass

# CHECK: lit.func @"call_generic{{.*}}"<[[DT:.*_dt]][dt]: !DType>()
fn call_generic[dt: DType]():
  # CHECK: %[[C57:.*]] = {{.*}}constant{{.*}} 57
  # CHECK: lit.call @"$parameters"::@"generic_fn{{.*}}"<:!DType [[DT]], :!Int {{.*}}42{{.*}}, :type !DType>(%[[C57]])
  generic_fn[dt, 42, DType](57)

  # CHECK: %[[C57_2:.*]] = {{.*}}constant{{.*}} 57
  # CHECK: lit.call @"$parameters"::@"generic_fn{{.*}}"<:!DType [[DT]], :!Int #lit.struct<{value = 13}>, :type @"$parameters"::@OurSIMD<:!Int #lit.struct<{value = 4}>{{.*}}, :!DType [[DT]]>{{.*}}>(%[[C57_2]])
  generic_fn[dt, 13, OurSIMD[4, dt]](57)

# CHECK-LABEL: lit.struct.decl @TestParamStruct<
# CHECK-SAME: [[A:.*]][A]: !Int>
@value
@register_passable
struct TestParamStruct[A: Int]:

  # CHECK: lit.func @"method{{.*}}<[[B:.*_B]][B]: !Int>(%self: !kgen.declref<{{.*}}TestParamStruct<:!Int [[A]]>{{.*}}> borrow,
  # CHECK-SAME: %other: {{.*}}@TestParamStruct<:!Int apply({{.*}}__add__{{.*}}, [[A]], [[B]])>{{.*}}> borrow)
  fn method[B: Int](self: TestParamStruct[A], other: TestParamStruct[A+B]):
    pass

  # CHECK-LABEL: lit.func @"aliases{{.*}}%x: {{.*}}@TestParamStruct<
  fn aliases(self, x: TestParamStruct[TestParamStruct[A].TypeLevelAlias]):
    # CHECK: lit.alias.decl [[B:.*]]: !Int = <apply({{.*}}__add__{{.*}}, apply({{.*}}__mul__{{.*}}, [[A]], [[A]]), {{.*}}1{{.*}})>
    alias B = A*A+1
    # CHECK: lit.alias.decl [[C:.*]]: !Int = <apply({{.*}}__mul__{{.*}}, [[B]], [[A]])>
    alias C = B*A
    # CHECK: lit.alias.decl [[D:.*]]: {{.*}}@TestParamStruct<:!Int {{.*}}1{{.*}}> = <apply(:!lit.signature<() ownedresult -> {{.*}}@TestParamStruct<:!Int {{.*}}1{{.*}}>>> {{.*}}__init__()"<:!Int {{.*}}1
    alias D = TestParamStruct[1]()
    # CHECK: %temp = lit.varlet.decl {{.*}} : {{.*}}@TestParamStruct<:!Int [[C]]>
    var temp: TestParamStruct[C]

    # CHECK: lit.alias.decl {{.*}}intVal: !Int = <#lit.struct<{value = 42}>>
    alias intVal : Int = 42

    # CHECK: %temp2 = lit.varlet.decl {{.*}} : {{.*}}@TestParamStruct<:!Int apply({{.*}}__mul__{{.*}}, [[A]], [[A]])
    var temp2: TestParamStruct[TestParamStruct[A].TypeLevelAlias]

  # CHECK: lit.alias.decl {{.*}}TypeLevelAlias: !Int = <apply({{.*}}__mul__{{.*}}, [[A]], [[A]])
  alias TypeLevelAlias = A*A

# Test that we support partially bound parameters.
fn testTestParamStruct(a: TestParamStruct[4]):
  # CHECK: %arg11 = lit.varlet.decl {{.*}} : {{.*}}@TestParamStruct<:!Int {{.*}}11
  # CHECK: %0 = lit.call {{.*}}@TestParamStruct::@"__init__{{.*}}<:!Int {{.*}}11{{.*}}>()
  # CHECK: lit.ref.store %0, %arg11
  var arg11 = TestParamStruct[11]()

  # CHECK: %1 = lit.ref.load %arg11
  # CHECK: lit.call {{.*}}@TestParamStruct::@"method{{.*}}<{{.*}}4{{.*}}7{{.*}}>(%a, %2)
  a.method[7](arg11)

# CHECK-LABEL: lit.func @"testSIMD(
fn testSIMD(a: Float64,
            b: Int32,
            inout reff: SIMD[DType.int32, 1]):
  # CHECK: %field1 = lit.varlet.decl {{.*}} : !lit.ref<scalar<f64>,
  var field1 = a.value
  # CHECK: %field2 = lit.varlet.decl {{.*}} : !lit.ref<scalar<si32>,
  var field2 = reff.value

  # Test calls to methods and operators on parameterized type.
  _ = a.fma(a, a)
  _ = b.fma(b, b)
  # CHECK: lit.call {{.*}}@"$builtin"::@"$simd"::@SIMD::@"__add__({{.*}}<:{{.*}} dtype = f64{{.*}}, {{.*}} = 1{{.*}}>(%a, %a)
  var x = a+a
  # CHECK: lit.call {{.*}}@"$builtin"::@"$simd"::@SIMD::@"__add__({{.*}}<:{{.*}} dtype = si32{{.*}}, {{.*}} = 1{{.*}}>(%b, %b)
  var y = b+b

# Show that forward references of parameter names can be correctly resolved.
#
# CHECK-LABEL: lit.func @"paramResolution[
# CHECK-SAME: $int::Int,
# CHECK-SAME: $parameters::StructWithIntParam[*(0,0)],
# CHECK-SAME: $int::Int,
# CHECK-SAME: $parameters::StructWithIntParam[*(0,2)]
# CHECK-SAME: ]()"<
# CHECK-SAME: [[SIZE1:.*_size1]][size1]: !Int, {{.*}}[a]: @"$parameters"::@StructWithIntParam<:!Int [[SIZE1]]> :{{.*}}>,
# CHECK-SAME: [[SIZE2:.*_size2]][size2]: !Int, {{.*}}[b]: @"$parameters"::@StructWithIntParam<:!Int [[SIZE2]]> :{{.*}}>>()
fn paramResolution[size1: Int, a: StructWithIntParam[size1],
                   size2: Int, b: StructWithIntParam[size2]]():
  pass

# Show that we can implicitly convert from 42's literal type to Int.
# CHECK-LABEL: lit.func @"implConversion
# CHECK: <{{.*}}[a]: @"$parameters"::@StructWithIntParam<:!Int #lit.struct<{{.*}}42}>>
fn implConversion[a: StructWithIntParam[42]]():
  pass

# CHECK-LABEL: lit.struct.decl @Pair<
# CHECK-SAME: [[DT:.*]][dt]: !DType>
@register_passable
struct Pair[dt: DType]:
 # CHECK: lit.struct.field a : {{.*}}@OurSIMD<:!Int {{.*}}42{{.*}}, :!DType [[DT]]>{{.*}}>
 # CHECK: lit.struct.field b : !Int
  var a : OurSIMD[42, dt]
  var b : Int

  # CHECK: lit.func @"__init__{{.*}} -> {{.*}}@Pair<:!DType [[DT]]>{{.*}}> attributes {{.*}}isStatic
  fn __init__(a: OurSIMD[42, dt]) -> Pair[dt]:
    # CHECK: [[TMP:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%a)
    # CHECK: %1 = kgen.param.constant: !Int {{.*}} 4
    # CHECK: %2 = lit.struct.create(a=%0, b=%1) : ({{.*}}@OurSIMD<:!Int #lit.struct<{value = 42}>, :!DType [[DT]]>{{.*}}>, !Int) -> {{.*}}@Pair<:!DType [[DT]]>
    return Pair[dt]{a: a, b: 4}
  # CHECK: }

  fn __copyinit__(self) -> Self: pass

# CHECK: }

# CHECK: useParameterizedField
fn useParameterizedField[x: Pair[DType.float32]]():
  # CHECK: lit.alias.decl {{.*}}y:
  alias y : OurSIMD[42, DType.float32] = x.a


# CHECK-LABEL: lit.func @"makePair
fn makePair(a: OurSIMD[42, DType.float32], b: Int) -> Pair[DType.float32]:
  # CHECK: [[TMP1:%.*]] = lit.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK:  = lit.struct.create(a=[[TMP1]], b=%b)
  return Pair[DType.float32]{a: a, b: b}

# CHECK-LABEL: lit.struct.decl @TypeParameter
# CHECK-SAME: <[[TYPE:.*]][T]: type>
struct TypeParameter[T: __mlir_type.`!kgen.type`]:
  # CHECK: @"bar($parameters::TypeParameter{{.*}}(%self: {{.*}} borrow_in_mem, %val: !kgen.paramref<[[TYPE]]> borrow)
  fn bar(self, val: T):
    pass

# Test that parameter decls can refine subsequent ones in the same param list.
# CHECK-LABEL: lit.struct.decl @ParamSubst
# CHECK-SAME: <[[TYPE:.*]][T]: type, [[SH:.*]][shape]: variadic<[[TYPE]]>>
struct ParamSubst[
    T: AnyRegType,
    shape: __mlir_type[`!kgen.variadic<`, T,`>`],
  ]: pass

# CHECK-LABEL: lit.func @"testParamSubst
fn testParamSubst():
  # CHECK: %xx = lit.varlet.decl {{.*}} : !lit.ref<@"$parameters"::@ParamSubst<:type index, :variadic<index> [1, 2]>
  var xx : ParamSubst[index, __mlir_attr.`#kgen.variadic<1, 2> : !kgen.variadic<index>`]


# Test parameter substitution.
# CHECK-LABEL: lit.func @"fnToCall{{.*}}"<
# CHECK-SAME: [[SIZE:.*_size]][size], {{.*}}[arr]: array<[[SIZE]], f32>>()
fn fnToCall[size: index, arr: __mlir_type[`!pop.array<`, size, `, f32>`]]():
  pass

# CHECK: lit.func @"fnWithCall{{.*}}"<
# CHECK-SAME: [[ARR:.*_array]][array]: array<10, f32>
fn fnWithCall[array: __mlir_type[`!pop.array<10, f32>`]]():
   # CHECK: lit.call @"$parameters"::@"fnToCall{{.*}}"<10, :array<10, f32> [[ARR]]>()
   fnToCall[Int(10).value, array]()

# CHECK-LABEL: lit.func @"meta_str{{.*}}"<{{.*}}[type]: !StringLiteral>() -> !kgen.none
fn meta_str[type: StringLiteral]():
  pass

# CHECK-LABEL: lit.func @"str_input_param()"() -> !kgen.none
fn str_input_param():
  # CHECK: %0 = lit.call @"$parameters"::@"meta_str{{.*}}"<:!StringLiteral {{.*}}"123"{{.*}}>()
  meta_str["123"]()

@value
@register_passable
struct TwoParams[a: Int, b: Int]:
    pass

# CHECK-LABEL: lit.func @"signature_capture
# CHECK-SAME: {{.*}}[a]: {{.*}}Int, {{.*}}[f]: !lit.signature<<"b": !Int>() ownedresult ->
# CHECK-SAME: {{.*}}TwoParams<:!Int {{.*}}a, :!Int *(0,0)>{{.*}}>
fn signature_capture[a: Int, f: fn[b: Int]() -> TwoParams[a, b]]():
    _ = f[2]()

# CHECK-LABEL: lit.func @"my_constrained{{.*}}()"
# CHECK-SAME: <[[COND:.*_cond]][cond]: !Bool, [[MESSAGE:.*_message]][message]: !StringLiteral>
fn my_constrained[cond: Bool, message: StringLiteral]():
    # CHECK: kgen.param.assert <apply({{.*}}__mlir_i1__{{.*}}, [[COND]])>, #lit.struct.extract<{{.*}}[[MESSAGE]], "value">
    __mlir_op.`kgen.param.assert`[cond=cond.__mlir_i1__(), message=message.value]()
    return


# CHECK-LABEL: lit.func @"pass_str_param
fn pass_str_param():
    # CHECK: lit.call {{.+}}my_constrained{{.*}}"<{{.*}}true{{.*}}, :!StringLiteral {{.*}}"foo"{{.*}}>()
    my_constrained[1==1, "foo"]()

# CHECK-LABEL: lit.func @"implicit_params
# CHECK-SAME: <?, [[VALUE0:.*]]: !Int, [[VALUE1:.*]]: !Int>
# CHECK-SAME: %value: {{.*}}@TwoParams<:!Int [[VALUE0]], :!Int [[VALUE1]]>
fn implicit_params(value: TwoParams):
    pass

# CHECK-LABEL: lit.func @"implicit_params_with_others
# CHECK-SAME: <{{.*}}a[a]: !Int, ?, [[LHS0:.*]]: !Int, [[LHS1:.*]]: !Int, [[RHS0:.*]]: !Int, [[RHS1:.*]]: !Int>
# CHECK-SAME: %lhs: {{.*}}@TwoParams<:!Int [[LHS0]], :!Int [[LHS1]]>
# CHECK-SAME: %rhs: {{.*}}@TwoParams<:!Int [[RHS0]], :!Int [[RHS1]]>
fn implicit_params_with_others[a: Int](lhs: TwoParams, rhs: TwoParams):
    pass

# CHECK-LABEL: lit.func @"infer_implicit_params()"
fn infer_implicit_params():
    # CHECK: call {{.*}}implicit_params{{.*}}<:!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 2}>
    let one = TwoParams[1, 2]()
    implicit_params(one)
    let two = TwoParams[3, 4]()
    # CHECK: call {{.*}}implicit_params_with_others{{.*}}<:!Int #lit.struct<{value = 42}>,
    # CHECK-SAME: :!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 3}>, :!Int #lit.struct<{value = 4}>>
    implicit_params_with_others[42](one, two)

fn implicit_params_with_var_params[*Ts: Int](s: TwoParams[1]): pass

# CHECK-LABEL: lit.func @"test_implicit_params_with_var_params
fn test_implicit_params_with_var_params():
    # CHECK: %[[VAL0:.*]] = lit.call @{{.*}}::@TwoParams::@"__init__()"<:!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 2}>>()
    # CHECK: call @{{.*}}::@"implicit_params_with_var_params{{.*}}"<:variadic<!Int> [], :!Int #lit.struct<{value = 2}>>(%[[VAL0]])
    implicit_params_with_var_params(TwoParams[1, 2]())

# CHECK-LABEL: lit.func @"explicit_autoparameterization
# CHECK-SAME: "<?, [[V0:.*_v0]]: !Int, [[W0:.*_w0]]: !Int, [[W1:.*_w1]]: !Int>(
# CHECK-SAME: %v: {{.*}}::@TwoParams<:!Int #lit.struct<{value = 5}>, :!Int [[V0]]>, !lit.metatype<@{{.*}}::@TwoParams<:!Int #lit.struct<{value = 5}>, :!Int [[V0]]>>
# CHECK-SAME: %w: {{.*}}::@TwoParams<:!Int [[W0]], :!Int [[W1]]>, !lit.metatype<@{{.*}}::@TwoParams<:!Int [[W0]], :!Int [[W1]]>>
fn explicit_autoparameterization(v: TwoParams[5, _], w: TwoParams[b=_, a=_]):
    pass

@register_passable("trivial")
struct IndexParam[x: index]:
  pass


# CHECK-LABEL: lit.func @"auto_kw_default
# CHECK-SAME: <[[U:.*]][u] = 3, |, [[V:.*]][v] = 3, ?, [[A:.*]], [[B:.*]]>(%a
fn auto_kw_default[u: index = index_three, /, v: index = index_three](a: IndexParam, b: IndexParam):
  pass


# CHECK-LABEL: lit.func @"test_auto_kw_default
# CHECK-SAME: <?, [[A:.*]], [[B:.*]]>(%a
fn test_auto_kw_default(a: IndexParam, b: IndexParam):
  # CHECK-NEXT: <3, 3, [[A]], [[B]]>
  auto_kw_default(a, b)
  # CHECK-NEXT: <1, 3, [[A]], [[B]]>
  auto_kw_default[index_one](a, b)
  # CHECK-NEXT: <3, 2, [[A]], [[B]]>
  auto_kw_default[v=index_two](a, b)
  # CHECK-NEXT: <1, 2, [[A]], [[B]]>
  auto_kw_default[index_one, v=index_two](a, b)


##===----------------------------------------------------------------------===##
# Memory-only parameters
##===----------------------------------------------------------------------===##

@value
struct MemoryType:
    var value: Int

struct NonMovableMemoryType:
    var value: Int

    fn __init__(inout self, value: Int):
        self.value = value

fn makeMemoryValue(x: Int) -> MemoryType:
    return x

fn passMemoryValue(x: MemoryType) -> MemoryType:
    return x

# CHECK-LABEL: lit.func @"callMemoryValueParam
fn callMemoryValueParam():
    # CHECK: paramValue: {{.*}}MemoryType = <apply_result_slot({{.*}}makeMemoryValue{{.*}}, {{.*}}1234
    alias paramValue = makeMemoryValue(1234)
    # CHECK: %dynamicLet = lit.varlet.decl
    # CHECK: %[[PARAM_VALUE:.*]] = kgen.param.materialize: !MemoryType = <{{.*}}paramValue>
    # CHECK: lit.ref.store %[[PARAM_VALUE]], %dynamicLet
    let dynamicLet = paramValue

    alias nonMovable = NonMovableMemoryType(42)
    # CHECK: %dynamicVar = lit.varlet.decl
    # CHECK: %[[NON_MOVABLE:.*]] = kgen.param.materialize: !NonMovableMemoryType
    # CHECK: lit.ref.store %[[NON_MOVABLE]], %dynamicVar
    var dynamicVar = nonMovable

    # CHECK: copy: {{.*}}MemoryType = <apply_result_slot({{.*}}passMemoryValue{{.*}} store_to_mem({{.*}}paramValue
    alias copy = passMemoryValue(paramValue)
    # CHECK: lit.varlet.decl
    # CHECK: [[MVALUE:%.*]] = lit.varlet.decl "anonymous*"
    # CHECK: [[PVALUE:%.*]] = kgen.param.materialize: !MemoryType = <{{.*}}copy>
    # CHECK: lit.ref.store [[PVALUE]], [[MVALUE]]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut %anonymous2A_0
    # CHECK: call {{.*}}passMemoryValue{{.*}}(%{{.*}}, [[IMMREF]])
    _ = passMemoryValue(copy)

    # CHECK: call {{.*}}memoryParam{{.*}}<:!MemoryType apply_result_slot({{.*}}__init__{{.*}}value = 22
    memoryParam[MemoryType(22)]()

# CHECK-LABEL: lit.func @"memoryParam
# CHECK-SAME: <{{.*}}[value]: !MemoryType>()
fn memoryParam[value: MemoryType]():
    pass

##===----------------------------------------------------------------------===##
# First-class functions as parameters.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"takeCallable{{.*}}"<
# CHECK-SAME: [[CALLABLE:.*_callable]][callable]: !lit.signature<(index borrow, |) -> index>>(%a: index borrow) -> index
fn takeCallable[
     callable: fn(index) -> index
   ](a: index) -> index:
  # CHECK-NEXT: %0 = lit.call_param[!lit.signature<(index borrow, |) -> index>: [[CALLABLE]]](%a)
  # CHECK-NEXT: lit.return %0
  return callable(a)

fn takeAndReturnIndex(x: index) -> index:
  return x

# CHECK-LABEL: lit.func @"takeAndReturnIndex
fn passFunction(a: index) -> index:
  # CHECK: lit.call @"$parameters"::@"takeCallable{{.*}}<:!lit.signature<(index borrow, |) -> index>
  # CHECK-SAME: rebind(:!lit.signature<("x": index borrow) -> index> @"$parameters"::@"takeAndReturnIndex{{.*}}")>(%a)
  return takeCallable[takeAndReturnIndex](a)

# CHECK-LABEL: lit.func @"callableWithParam{{.*}}"<{{.*}}[type]: dtype>() -> !kgen.none
fn callableWithParam[type: __mlir_type.`!kgen.dtype`]():
  pass

# CHECK-LABEL: lit.func @"takeCallable2
fn takeCallable2[
      func: fn[dt: __mlir_type.`!kgen.dtype`]() -> None
  ]():
      pass

# CHECK-LABEL: lit.func @"passFunctionParam2
fn passFunctionParam2():
  # CHECK: lit.call @"$parameters"::@"takeCallable2{{.*}}"<
  # CHECK-SAME: :!lit.signature<<"dt": dtype>() -> !kgen.none> rebind(:!lit.signature<<"type": dtype>() -> !kgen.none> @"$parameters"::@"callableWithParam
  takeCallable2[callableWithParam]()


@register_passable("trivial")
struct ParamType[x: index]:
    pass


# CHECK-LABEL: lit.func @"dependent_function_type
fn dependent_function_type[a: index, f: fn (ParamType[a]) -> None]():
    alias func = dependent_function_type
    # CHECK: bind_signature
    func[a, f]()


##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl {{.*}}boolDtype: dtype = <bool>
alias boolDtype = __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
# CHECK: lit.alias.decl {{.*}}FOURTY_TWO: !IntLiteral = <{{.*}}42
alias FOURTY_TWO = 42

# CHECK-LABEL: lit.struct.decl @A
# CHECK-SAME: <[[V:.*]][v]: !Int>
struct A[v: Int]:
  # CHECK: lit.alias.decl {{.*}}member: !Int = <apply({{.*}}__add__{{.*}}, [[V]], {{.*}}42
  alias member = v + FOURTY_TWO

# CHECK-LABEL: lit.func @"testUseOfAliases
fn testUseOfAliases(a: Bool):
  # This type checks.
  _ = SIMD[DType(boolDtype), 4].splat(a)
  # CHECK: lit.alias.decl {{.*}}y: !Int = <{{.*}}44
  alias y = A[2].member

@register_passable
struct MyDType:
  var state : __mlir_type.index

  fn __copyinit__(self) -> Self:
    return Self{state: self.state}

  fn __init__(value: __mlir_type.index) -> MyDType:
     return MyDType{state: value}

  fn __eq__(self, rhs: MyDType) -> Bool:
     return True  # TODO: buggy impl :-)

  alias ui8 = MyDType(Int(1).value)
  alias float32 = MyDType(Int(2).value)
  alias float64 = MyDType(Int(3).value)

  # CHECK: lit.alias.decl {{.*}}ui16: !MyDType = <#lit.struct<{state = 7}>>
  alias ui16 = MyDType{state: Int(7).value}

struct MyVector[size: Int, dtype: MyDType]:
  pass

fn testMyDType[dt: MyDType](a: MyVector[4, MyDType.float32],
                            b: MyVector[4, dt]):

   constrained[dt == MyDType.float64]()

# Issue #6828: Unqualified name lookup into structs doesn't work
# CHECK-LABEL: lit.struct.decl @UnqualAliasLookup
# CHECK-SAME: <[[PARAM:.*]][param]: !Int>
struct UnqualAliasLookup[param: Int]:
  # CHECK: lit.alias.decl {{.*}}member: !Int = <apply({{.*}}__add__{{.*}}, [[PARAM]], {{.*}}1{{.*}})>
  alias member = param+1
  fn get(self) -> Int:
    # CHECK: %0 = kgen.param.constant: !Int = <apply({{.*}}__add__{{.*}}, [[PARAM]], {{.*}}1{{.*}})>
    return Self.member

##===----------------------------------------------------------------------===##
# Variadic parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"fnWithVariadics{{.*}}()"<{{.*}}[b]: variadic<!Int>>
fn fnWithVariadics[*b: Int]():
  pass

# CHECK-LABEL: lit.struct.decl @StructWithVariadics
# CHECK-SAME: <[[B:.*]][b]: variadic<!Int>>
struct StructWithVariadics[*b: Int]:
    fn __init__(inout self, i: Int):
        pass

# CHECK-LABEL: lit.func @"useParamVariadics
fn useParamVariadics():
  # CHECK-NEXT: lit.call @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<!Int> []>()
  fnWithVariadics()

  # CHECK: lit.call @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<!Int> [#lit.struct<{value = 1}>]>()
  fnWithVariadics[1]()
  # CHECK: lit.call @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<!Int> [#lit.struct<{value = 1}>, #lit.struct<{value = 2}>]>()
  fnWithVariadics[1, 2]()

  # This keeps the parameters unbound, allowing them to be used with different length..
  # CHECK-NEXT: lit.alias.decl {{.*}}fnAlias: !lit.signature<<"b": variadic<!Int>>() param_vararg -> !kgen.none>
  # CHECK-SAME: = <@"$parameters"::@"fnWithVariadics{{.*}}">
  alias fnAlias = fnWithVariadics

  # Use of an unbound thing in a DRValue context binds an empty variadic list.
  # FIXME(#29495): Pack references aren't working right.
  # HECK-NEXT: [[TMP:%.*]] = kgen.create_closure[!lit.signature<() -> !kgen.none>: @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<!Int> []>]()
  # HECK-NEXT: %fnLet = lit.varlet.decl "fnLet" : {{.*}}!kgen.signature<!lit.signature<() -> !kgen.none>>
  # HECK-NEXT: lit.ref.store [[TMP]], %fnLet
  # var fnLet = fnWithVariadics

  # CHECK-NEXT: %a = lit.varlet.decl {{.*}} : !lit.ref<@"{{.*}}::@StructWithVariadics<:variadic<!Int> []>
  var a: StructWithVariadics
  # CHECK-NEXT: %b = lit.varlet.decl {{.*}} : !lit.ref<@{{.*}}::@StructWithVariadics<:variadic<!Int> [#lit.struct<{value = 1}>]>
  var b: StructWithVariadics[1]
  # CHECK-NEXT: %c = lit.varlet.decl {{.*}} : !lit.ref<@{{.*}}::@StructWithVariadics<:variadic<!Int> [#lit.struct<{value = 1}>, #lit.struct<{value = 2}>]>
  var c: StructWithVariadics[1, 2]

  # TODO(16040): fix symbol name mangling to erase parameter name 'b'
  # CHECK: lit.call {{.*}}@StructWithVariadics::@"__init__(${{.*}}::StructWithVariadics[b]=&,{{.*}}$int::Int)"{{.*}}<:variadic<!Int> [#lit.struct<{value = 1}>]>
  var d = StructWithVariadics[1](2)
  # CHECK: lit.call {{.*}}@StructWithVariadics::@"__init__(${{.*}}::StructWithVariadics[b]=&,{{.*}}$int::Int)"{{.*}}<:variadic<!Int> []>
  var e = StructWithVariadics(3)


# CHECK-LABEL: lit.func @"variadic_parameter{{.*}}"<{{.*}}[elems]: variadic<index>>
fn variadic_parameter[elems: __mlir_type.`!kgen.variadic<index>`]() -> Int:
    return 3

fn dependent_variadic_parameter[
    type: __mlir_type.`!kgen.type`, *values: type
](): pass

# CHECK-LABEL: lit.func @"pass_variadic{{.*}}"<
# CHECK-SAME: [[ELEMS:.*_elems]][elems]: variadic<index>>
fn pass_variadic[elems: __mlir_type.`!kgen.variadic<index>`]():
    # CHECK-NEXT: lit.call @"$parameters"::@"variadic_parameter{{.*}}"<:variadic<index> [[ELEMS]]>
    _ = variadic_parameter[elems]()
    # CHECK: lit.call @"$parameters"::@"dependent_variadic_parameter{{.*}}"<:type !Int, :variadic<!Int>
    _ = dependent_variadic_parameter[Int, 1, 2]()


##===----------------------------------------------------------------------===##
# Parameter Inference
##===----------------------------------------------------------------------===##

@register_passable("trivial")
struct StaticVec[size: Int]:
  fn __init__[type: __mlir_type.`!kgen.dtype`](v: __mlir_type[`!pop.simd<`, size.value, `, `, type, `>`]) -> StaticVec[size]:
      return Self{}

  @staticmethod
  fn thing[type: __mlir_type.`!kgen.dtype`](v: __mlir_type[`!pop.simd<`, size.value, `, `, type, `>`]):
      return

fn callee1[size: Int](v: StaticVec[size]): pass
fn callee2[T: __mlir_type.`!kgen.type`](v: T): pass
fn callee3[size: __mlir_type.index, type: __mlir_type.`!kgen.dtype`]
   (v:  __mlir_type[`!pop.simd<`, size, `, `, type, `>`]): pass
fn callee4[T: __mlir_type.`!kgen.type`]
   (v:  __mlir_type[`!kgen.pointer<`, T, `>`]): pass

# CHECK-LABEL: lit.func @"testParamInference{{.*}}"<
# CHECK-SAME: [[SIZE:.*_size]][size]: !Int>(
fn testParamInference[size: Int](a: StaticVec[4], b: StaticVec[size],
                                 b2: StaticVec[size*2],
                                 c: __mlir_type.`!pop.simd<17, f32>`,
                                 d: __mlir_type.`!kgen.pointer<f32>`):
  # CHECK-NEXT: lit.call @{{.*}}callee1{{.*}}<{{.*}}4{{.*}}>(%a)
  callee1(a)
  # CHECK-NEXT: lit.call @{{.*}}callee1{{.*}}<:!Int [[SIZE]]>(%b)
  callee1(b)
  # CHECK-NEXT: lit.call @{{.*}}callee1{{.*}}<:!Int apply({{.*}}__mul__{{.*}}, [[SIZE]], {{.*}}2{{.*}})>(%b2)
  callee1(b2)
  # CHECK-NEXT: lit.call @{{.*}}callee2{{.*}}<:type @"$parameters"::@StaticVec<:!Int [[SIZE]]>{{.*}}>(%b)
  callee2(b)
  # CHECK-NEXT: lit.call @{{.*}}callee3{{.*}}<17, :dtype f32>(%c)
  callee3(c)
  # CHECK-NEXT: lit.call @{{.*}}callee4{{.*}}<:type f32>(%d)
  callee4(d)

# CHECK-LABEL: lit.struct.decl @Abstraction
# CHECK-SAMEL <[[A:.*]]: !Int>
@value
@register_passable
struct Abstraction[a: Int]:
  alias val = a.value

  @staticmethod
  fn push[b: Int]() -> Abstraction[a + b]:
      return Abstraction[a + b]()

  @staticmethod
  fn pull[b: Int](value: Abstraction[a + b]):
      return

# CHECK-LABEL: lit.func @"testDependentType{{.*}}"<
# CHECK-SAME: [[RANK:.*_rank]][rank]: !Int, {{.*}}[shape]: array<#lit.struct.extract<:!Int [[RANK]], "value">
fn testDependentType[
    rank: Int,
    shape: __mlir_type[`!pop.array<`, rank.value, `, index>`],
]():
    pass

# CHECK-LABEL: lit.func @"testParameterEvaluator()"
fn testParameterEvaluator():
  # CHECK-NEXT: lit.alias.decl {{.*}}x = <1>
  alias x = Abstraction[1].val
  # CHECK-NEXT: %y = lit.varlet.decl "y"
  # CHECK-NEXT: %0 = lit.call @"$parameters"::@Abstraction::@"push{{.*}}"<:{{.*}} = 1{{.*}}, :{{.*}} = 2{{.*}}>()
  # CHECK-NEXT: %1 = kgen.rebind %0 : {{.*}} to {{.*}}@Abstraction<:!Int {{.*}} = 3}
  # CHECK-NEXT: lit.ref.store %1, %y
  var y : Abstraction[3] = Abstraction[1].push[2]()
  # CHECK-NEXT: [[Y:%.*]] = lit.ref.load %y
  # CHECK-NEXT: [[RB:%.*]] = kgen.rebind [[Y]] : {{.*}}@Abstraction<:!Int {{.*}} = 3}
  # CHECK-NEXT: lit.call {{.*}}@Abstraction::@"pull{{.*}}"<{{.*}}>([[RB]])
  Abstraction[1].pull[2](y)
  # CHECK-NEXT: lit.call {{.*}}@"testDependentType{{.*}}"<:{{.*}} = 1{{.*}}, :array<1, index>
  testDependentType[1, __mlir_attr.`#pop.array<0> : !pop.array<1, index>`]()


fn takeAbstraction2(value: Abstraction[2]):
    return

@register_passable
struct AnotherAbstraction[a: Int]:
    var value : Abstraction[a + 1]

    fn __init__() -> Self:
        return Self{value: Abstraction[a + 1]()}

    fn __copyinit__(self) -> Self:
        return Self{value: self.value}

# CHECK-LABEL: lit.func @"testDependentField()"
fn testDependentField():
    var lvalue = AnotherAbstraction[1]()
    # CHECK: [[VALUE_PTR:%.*]] = lit.ref.struct.ger %lvalue[value]
    # CHECK-NEXT: kgen.rebind [[VALUE_PTR]] {{.*}} to
    # CHECK-SAME: !lit.ref<{{.*}}@Abstraction<:!Int {{.*}} 2}>>
    takeAbstraction2(lvalue.value)


fn tail_types[T: AnyRegType, *U: AnyRegType](a: T, *b: *U):
    pass

# CHECK-LABEL: lit.func @"call_with_tail_types()"
fn call_with_tail_types():
    # CHECK: call {{.*}}tail_types{{.*}}<:type !Int, :variadic<type> []>
    tail_types(1)
    # CHECK: call {{.*}}tail_types{{.*}}<:type !Int, :variadic<type> [{{.*}}FloatLiteral]>
    tail_types(1, 1.2)
    # CHECK: call {{.*}}tail_types{{.*}}<:type !Int, :variadic<type> [{{.*}}Int]>
    tail_types(1, 77)

# COM: We can't infer parameters from the default value, but we need to test if
# COM: if other parameters are inferred correctly in their presence.
fn infer_with_default_arg[T: AnyRegType](a: T, b: Int = 7):
    pass

# CHECK-LABEL: lit.func @"test_infer_with_default_arg()"
fn test_infer_with_default_arg():
    # lit.call @{{.*}}::@"infer_with_default_arg[AnyRegType]($0,{{.*}}::Int)"<:type !Int>
    infer_with_default_arg(128)

fn fn_with_param[x: Int](y: Abstraction[x]):
    pass

# CHECK-LABEL: lit.func @"indirect_call_infer_params
fn indirect_call_infer_params():
    alias callee = fn_with_param
    # CHECK: call_param[!lit.signature<("y": {{.*}}Abstraction<:!Int #lit.struct<{value = 2}>>
    # CHECK-SAME: bind_signature(:!lit.signature<<"x": !Int>("y": {{.*}}Abstraction<:!Int *(0,0)>
    # CHECK-SAME: callee, #lit.struct<{value = 2}>
    callee(Abstraction[2]())

##===----------------------------------------------------------------------===##
# Access parameter through structure
##===----------------------------------------------------------------------===##

struct MultiStruct[p1: Int, p2: Int, p3: Int]:
    fn __init__(inout self): pass

fn foo[x: Int]():
  pass

fn bar(x : Int):
  pass

# CHECK-LABEL: lit.func @"reference_params_through_struct
fn reference_params_through_struct():
    var x = MultiStruct[52, 9, 33]()

    # CHECK: %[[Y:.*]] = lit.varlet.decl "y"
    # CHECK-NEXT: %[[P:.*]] = kgen.param.constant: {{.*}} <#lit.struct<{value = 52}>
    # CHECK-NEXT: lit.ref.store %[[P]], %[[Y]]
    var y = x.p1

    # CHECK: %[[P:.*]] = kgen.param.constant: {{.*}} <#lit.struct<{value = 9}>
    # CHECK-NEXT: lit.call @"{{.*}}bar({{.*}})"(%[[P]])
    bar(x.p2)

    # CHECK: lit.call @{{.*}}foo{{.*}}<:!Int #lit.struct<{value = 33}>>
    foo[x.p3]()

    # CHECK: %[[Z:.*]] = lit.varlet.decl "z"
    # CHECK-NEXT: %[[P:.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 1}>>
    # CHECK-NEXT: lit.ref.store %[[P]], %[[Z]]
    var z = MultiStruct[1, 2, 3].p1

    # CHECK: %[[P:.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 2}>>
    # CHECK-NEXT: lit.call @"{{.*}}bar({{.*}})"(%[[P]])
    bar(MultiStruct[1, 2, 3].p2)

    # CHECK: lit.call @{{.*}}foo{{.*}}<:!Int #lit.struct<{value = 3}>>
    foo[MultiStruct[1, 2, 3].p3]()

##===----------------------------------------------------------------------===##
# Default function parameters
##===----------------------------------------------------------------------===##

fn default_params[a: Int, b: Int = 7, c: StringLiteral = "woof"]():
    pass


# CHECK-LABEL: lit.func @"test_default_params()"
fn test_default_params():
    # CHECK: lit.call @"{{.*}}@"default_params[{{.*}}::Int,{{.*}}::Int,{{.*}}::StringLiteral]()"
    # CHECK-SAME: <:!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 7}>, :!StringLiteral #lit.struct<{value: string = "woof"}>>
    default_params[1]()

    # CHECK: lit.call @"{{.*}}@"default_params[{{.*}}::Int,{{.*}}::Int,{{.*}}::StringLiteral]()"
    # CHECK-SAME: <:!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 8}>, :!StringLiteral #lit.struct<{value: string = "woof"}>>
    default_params[2, 8]()

    # CHECK: lit.call @"{{.*}}@"default_params[{{.*}}::Int,{{.*}}::Int,{{.*}}::StringLiteral]()"
    # CHECK-SAME: <:!Int #lit.struct<{value = 4}>, :!Int #lit.struct<{value = 9}>, :!StringLiteral #lit.struct<{value: string = "meow"}>>
    default_params[4, 9, "meow"]()


fn test_indirect_default_params():
    # CHECK: lit.alias.decl [[CALLEE:.*_callee]]: !lit.signature
    alias callee = default_params

    # CHECK: lit.call_param[!lit.signature<() -> !kgen.none>: bind_signature(:!lit.signature<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> [[CALLEE]],
    # CHECK-SAME: #lit.struct<{value = 1}>, #lit.struct<{value = 7}>, #lit.struct<{value: string = "woof"}>)]()
    callee[1]()

    # CHECK: lit.call_param[!lit.signature<() -> !kgen.none>: bind_signature(:!lit.signature<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> [[CALLEE]],
    # CHECK-SAME: #lit.struct<{value = 2}>, #lit.struct<{value = 8}>, #lit.struct<{value: string = "woof"}>)]()
    callee[2, 8]()

    # CHECK: lit.call_param[!lit.signature<() -> !kgen.none>: bind_signature(:!lit.signature<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> [[CALLEE]],
    # CHECK-SAME: #lit.struct<{value = 4}>, #lit.struct<{value = 9}>, #lit.struct<{value: string = "meow"}>)]()
    callee[4, 9, "meow"]()


# COM: check that inferred parameter values take precedence over defaults
# CHECK-LABEL: lit.func @"inferred_default_param
fn inferred_default_param[dt: DType, w: Int = 8](a: OurSIMD[w, dt]):
    pass


# CHECK: lit.func @"test_inferred_default_param{{.*}}"<[[X:.*_x]][x]: !Int>
# CHECK: lit.call @{{.*}}@"inferred_default_param[{{.*}}::DType,{{.*}}::Int]({{.*}}::OurSIMD[*(0,1), *(0,0)])"<:!DType #lit.struct<{value: dtype = f32}>, :!Int #lit.struct<{value = 4}>>
# CHECK: lit.call @{{.*}}@"inferred_default_param[{{.*}}::DType,{{.*}}::Int]({{.*}}::OurSIMD[*(0,1), *(0,0)])"<:!DType #lit.struct<{value: dtype = f32}>, :!Int [[X]]>
fn test_inferred_default_param[
    x: Int
](concrete: OurSIMD[4, DType.float32], p: OurSIMD[x, DType.float32]):
    inferred_default_param(concrete)
    inferred_default_param(p)


# COM: basic check for memory-only default parameters
@value
struct MemoryOnlyType:
    pass


# CHECK: lit.func @"mem_only_default_param[{{.*}}::MemoryOnlyType]()"<{{.*}}[x]: !MemoryOnlyType =
# CHECK-SAME: apply_result_slot(:(!lit.ref<!MemoryOnlyType, mut #lit.lifetime> init_self) -> !kgen.none
# CHECK-SAME: rebind(:!lit.signature<[1](!lit.ref<!MemoryOnlyType, mut *[0,0]> init_self, |) -> !kgen.none> @"$parameters"::@MemoryOnlyType::@"__init__($parameters::MemoryOnlyType=&)"))>(
fn mem_only_default_param[x: MemoryOnlyType = MemoryOnlyType()]():
    pass

# CHECK-LABEL: lit.func @"test_mem_only_default_param()"
# CHECK: lit.call @{{.*}}@"mem_only_default_param[{{.*}}::MemoryOnlyType]()"<
# CHECK-SAME: :!MemoryOnlyType apply_result_slot(:(!lit.ref<!MemoryOnlyType, mut #lit.lifetime> init_self) -> !kgen.none rebind(:!lit.signature<[1](!lit.ref<!MemoryOnlyType, mut *[0,0]> init_self, |) -> !kgen.none> @"$parameters"::@MemoryOnlyType::@"__init__($parameters::MemoryOnlyType=&)"))>
fn test_mem_only_default_param():
    mem_only_default_param()

# CHECK-LABEL: lit.func @"param_default{{.*}}"<
# CHECK-SAME: [[X:.*]][x]: !Int = #lit.struct<{value = 1}>>(%y: !Int borrow = [[X]])
fn param_default[x: Int = 1](y: Int = x):
    pass

# CHECK-LABEL: lit.func @"test_param_default
fn test_param_default():
    # CHECK: [[C:%.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 4}>>
    # CHECK-NEXT: call {{.*}}param_default{{.*}}<:!Int #lit.struct<{value = 4}>>([[C]]
    param_default[4]()
    # CHECK: [[C:%.*]] = kgen.param.constant: !Int = <#lit.struct<{value = 1}>>
    # CHECK-NEXT: call {{.*}}param_default{{.*}}<:!Int #lit.struct<{value = 1}>>([[C]]
    param_default()

##===----------------------------------------------------------------------===##
# Default struct parameters
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.decl @DefaultParams<{{.*}}: !Int, {{.*}}: !Int = #lit.struct<{value = 7}>, {{.*}}: !StringLiteral = #lit.struct<{value: string = "woof"}>
@value
struct DefaultParams[a: Int, b: Int = 7, msg: StringLiteral = "woof"]: pass

# CHECK-LABEL: lit.func @"test_default_param_struct()"
fn test_default_param_struct():
    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 7}>, :!StringLiteral #lit.struct<{value: string = "woof"}>
    alias T = DefaultParams[1]
    # CHECK-NEXT: %[[INIT:.*]] = lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@DefaultParams<
    # CHECK-SAME:   :!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 7}>, :!StringLiteral #lit.struct<{value: string = "woof"}>
    # CHECK-NEXT: lit.call @{{.*}}@DefaultParams::@"__init__({{.*}}::DefaultParams[a, b, msg]=&)"{{.*}}<:!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 7}>, :!StringLiteral #lit.struct<{value: string = "woof"}>
    _ = DefaultParams[1]()

    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 3}>, :!StringLiteral #lit.struct<{value: string = "woof"}>
    alias U = DefaultParams[2, 3]
    # CHECK-NEXT: %[[INIT:.*]] = lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@DefaultParams<
    # CHECK-SAME:   :!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 3}>, :!StringLiteral #lit.struct<{value: string = "woof"}>
    # CHECK-NEXT: lit.call @{{.*}}@DefaultParams::@"__init__({{.*}}::DefaultParams[a, b, msg]=&)"{{.*}}<:!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 3}>, :!StringLiteral #lit.struct<{value: string = "woof"}>
    _ = DefaultParams[2, 3]()

    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int #lit.struct<{value = 4}>, :!Int #lit.struct<{value = 5}>, :!StringLiteral #lit.struct<{value: string = "meow"}>
    alias S = DefaultParams[4, 5, "meow"]
    # CHECK-NEXT: %[[INIT:.*]] = lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@DefaultParams<
    # CHECK-SAME:   :!Int #lit.struct<{value = 4}>, :!Int #lit.struct<{value = 5}>, :!StringLiteral #lit.struct<{value: string = "meow"}>
    # CHECK-NEXT: lit.call @{{.*}}@DefaultParams::@"__init__({{.*}}::DefaultParams[a, b, msg]=&)"{{.*}}<:!Int #lit.struct<{value = 4}>, :!Int #lit.struct<{value = 5}>, :!StringLiteral #lit.struct<{value: string = "meow"}>
    _ = DefaultParams[4, 5, "meow"]()


# CHECK: lit.struct.decl @AllDefaultParams<{{.*}}: !Int = #lit.struct<{value = 0}>, {{.*}}: !MemoryOnlyType = apply_result_slot({{.*}}@MemoryOnlyType::@"__init__
@value
struct AllDefaultParams[x: Int = 0, v: MemoryOnlyType = MemoryOnlyType()]: pass

# CHECK-LABEL: lit.func @"test_default_param_struct_all_default()"
fn test_default_param_struct_all_default():
    # CHECK: lit.alias.decl {{.*}}T: metatype<{{.*}}@AllDefaultParams{{.*}}> = <@{{.*}}::@AllDefaultParams<
    # CHECK-SAME: :!Int #lit.struct<{value = 0}>,
    # CHECK-SAME: :!MemoryOnlyType apply_result_slot({{.*}}@MemoryOnlyType::@"__init__
    alias T = AllDefaultParams[]

    # CHECK: %[[INIT:.*]] = lit.varlet.decl {{.*}} : !lit.ref<@{{.*}}::@AllDefaultParams<
    # CHECK-SAME:   :!Int #lit.struct<{value = 0}>, :!MemoryOnlyType apply_result_slot({{.*}}@MemoryOnlyType::@"__init__
    # CHECK-NEXT: = lit.call @{{.*}}::@AllDefaultParams::@"__init__({{.*}}::AllDefaultParams[x, v]=&)"{{.*}}<:!Int #lit.struct<{value = 0}>, :!MemoryOnlyType
    _ = AllDefaultParams[]()


# COM: Issue #22763
fn IntForType[T: AnyRegType]() -> Int:
    return 1

struct StructWithParametricDefaultValue[T: AnyRegType, N: Int = IntForType[T]()]:
    pass

# CHECK-LABEL: lit.func @"test_struct_with_parametric_default_value()"
fn test_struct_with_parametric_default_value():
    # CHECK: lit.alias.decl {{.*}}_a: metatype<{{.*}}> = <@{{.*}}::@StructWithParametricDefaultValue<
    # CHECK-SAME: :type !Int
    # CHECK-SAME: :!Int apply(:!lit.signature<() -> !Int> @{{.*}}::@"IntForType[AnyRegType]()"{{.*}}<:type !Int>)>
    alias a = StructWithParametricDefaultValue[Int]


##===----------------------------------------------------------------------===##
# Function keyword parameters
##===----------------------------------------------------------------------===##

fn take_kw_params[a: Int, b: Int = 2, c: Int = 3](): pass

# CHECK-LABEL: lit.func @"test_simple_kw_params()"
fn test_simple_kw_params():
    # CHECK: lit.call @{{.*}}@"take_kw_params{{.*}}"<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 3}>>
    take_kw_params[5, b=7]()
    # CHECK: lit.call @{{.*}}@"take_kw_params{{.*}}"<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>>
    take_kw_params[5, b=7, c=9]()
    # CHECK: lit.call @{{.*}}@"take_kw_params{{.*}}"<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 9}>>
    take_kw_params[5, c=9]()
    # CHECK: lit.call @{{.*}}@"take_kw_params{{.*}}"<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>>
    take_kw_params[5, c=9, b=7]()
    # CHECK: lit.call @{{.*}}@"take_kw_params{{.*}}"<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>>
    take_kw_params[a=5, c=9, b=7]()
    # CHECK: lit.call @{{.*}}@"take_kw_params{{.*}}"<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>>
    take_kw_params[c=9, b=7, a=5]()


fn test_indirect_kw_params():
  # CHECK: lit.alias.decl [[CALLEE:.*]]: !lit.signature
  alias callee = take_kw_params
  # CHECK: lit.call_param[!lit.signature<() -> !kgen.none>: bind_signature(:!lit.signature<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> [[CALLEE]],
  # CHECK-SAME: #lit.struct<{value = 5}>, #lit.struct<{value = 2}>, #lit.struct<{value = 9}>)]()
  callee[5, c=9]()
  # CHECK: lit.call_param[!lit.signature<() -> !kgen.none>: bind_signature(:!lit.signature<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> [[CALLEE]],
  # CHECK-SAME: #lit.struct<{value = 5}>, #lit.struct<{value = 7}>, #lit.struct<{value = 9}>)]()
  callee[c=9, b=7, a=5]()


@register_passable("trivial")
struct MyInt:
    var value: Int

    @always_inline("nodebug")
    fn __init__(_a: Int) -> Self:
        return Self {value: _a}

fn overloaded_kw_param[a: Int, b: MyInt](): pass

fn overloaded_kw_param[a: Int, b: Int](): pass

# CHECK-LABEL: lit.func @"test_kw_params_overload
fn test_kw_params_overload[a: Int, b: Int]():
    # CHECK: lit.call @{{.*}}@"overloaded_kw_param[{{.*}}::Int,{{.*}}::Int]()"
    overloaded_kw_param[b=b, a=a]()
    # CHECK: lit.call @{{.*}}@"overloaded_kw_param[{{.*}}::Int,{{.*}}::MyInt]()"
    overloaded_kw_param[b = MyInt(b), a=a]()


##===----------------------------------------------------------------------===##
# Struct keyword parameters
##===----------------------------------------------------------------------===##

@value
struct KwParamStruct[a: Int, b: Int = 2, c: Int = 3]: pass

# CHECK-LABEL: lit.func @"test_struct_kw_params()"
fn test_struct_kw_params():
    # CHECK: lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 3}>
    _ = KwParamStruct[5, b=7]()
    # CHECK: lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>
    _ = KwParamStruct[5, b=7, c=9]()
    # CHECK: lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 9}>
    _ = KwParamStruct[5, c=9]()
    # CHECK: lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>
    _ = KwParamStruct[5, c=9, b=7]()
    # CHECK: lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>
    _ = KwParamStruct[a=5, c=9, b=7]()
    # CHECK: lit.varlet.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int #lit.struct<{value = 5}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 9}>
    _ = KwParamStruct[c=9, b=7, a=5]()

##===----------------------------------------------------------------------===##
# Partial binding
##===----------------------------------------------------------------------===##

@value
struct Thing[v: Int]: pass

struct CtadStruct[a: Int, b: Int]:
    fn __init__(inout self, x: Thing[a]): pass

    fn __init__(inout self, x: Thing[a], y: Thing[b]): pass

    @staticmethod
    fn foo(x: Thing[a]): pass

    @staticmethod
    fn foo(x: Thing[a], y: Thing[b]): pass

struct CtadStructWithDefault[a: Int, b: Int, c: Int = 8]:
    fn __init__(inout self, x: Thing[a]): pass

    fn __init__(inout self, x: Thing[a], y: Thing[b]): pass

    @staticmethod
    fn foo(x: Thing[a]): pass

    @staticmethod
    fn foo(x: Thing[a], y: Thing[b]): pass

# CHECK-LABEL: lit.func @"test_partial_binding_CTAD()"
fn test_partial_binding_CTAD():
    # CHECK: call @{{.*}}::@CtadStruct::@"__init__({{.*}})"{{.*}}<:!Int #lit.struct<{value = 6}>, :!Int #lit.struct<{value = 7}>>
    _ = CtadStruct[b=7](Thing[6]())
    # CHECK: call @{{.*}}::@CtadStruct::@"__init__({{.*}})"{{.*}}<:!Int #lit.struct<{value = 8}>, :!Int #lit.struct<{value = 9}>>
    _ = CtadStruct[](Thing[8](), Thing[9]())
    # CHECK: call @{{.*}}::@CtadStruct::@"foo({{.*}}<:!Int #lit.struct<{value = 6}>, :!Int #lit.struct<{value = 7}>>
    CtadStruct[b=7].foo(Thing[6]())
    # CHECK: call @{{.*}}::@CtadStruct::@"foo({{.*}}<:!Int #lit.struct<{value = 8}>, :!Int #lit.struct<{value = 9}>>
    CtadStruct[].foo(Thing[8](), Thing[9]())

    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int #lit.struct<{value = 6}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 8}>>
    _ = CtadStructWithDefault[b=7](Thing[6]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 8}>>
    _ = CtadStructWithDefault[](y=Thing[1](), x=Thing[2]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int #lit.struct<{value = 6}>, :!Int #lit.struct<{value = 9}>, :!Int #lit.struct<{value = 8}>>
    _ = CtadStructWithDefault(Thing[6](), Thing[9]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int #lit.struct<{value = 6}>, :!Int #lit.struct<{value = 7}>, :!Int #lit.struct<{value = 8}>>
    CtadStructWithDefault[b=7].foo(Thing[6]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int #lit.struct<{value = 2}>, :!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 8}>>
    CtadStructWithDefault[].foo(y=Thing[1](), x=Thing[2]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int #lit.struct<{value = 4}>, :!Int #lit.struct<{value = 3}>, :!Int #lit.struct<{value = 8}>>
    CtadStructWithDefault.foo(y=Thing[3](), x=Thing[4]())


# COM: https://github.com/modularml/mojo/issues/1227
# COM: Ensure default parameters are rebound during CTAD.
@value
@register_passable("trivial")
struct DependentDefault[x: Int = 1, y: Int = x]:
    pass


# CHECK-LABEL: lit.func @"dependent_default_ctad
fn dependent_default_ctad():
    # CHECK-NEXT: value: {{.*}}@DependentDefault<:!Int #lit.struct<{value = 1}>, :!Int #lit.struct<{value = 1}>>
    alias value = DependentDefault()


# CHECK-LABEL: lit.func @"scalar_type
# CHECK-SAME: <[[dt:.*]][dt]: !DType>
fn scalar_type[dt: DType]():
    # CHECK: alias.decl [[T:.*_T]]: metatype<{{.*}}SIMD<:!DType [[dt]],
    alias T = Scalar[dt]

    #FIXME(29495): reenable.
    # https://github.com/modularml/modular/issues/29495
    # HECK: lit.varlet.decl "value" = %{{.*}} : !kgen.declref<{{.*}}@SIMD<:!DType [[dt]],
    #var value: T = 1
    # HECK: call {{.*}}<:!DType [[dt]], {{.*}}, :!DType [[dt]]>(%value)
    #_ = value.cast[dt]()


struct T: pass

# CHECK-LABEL: lit.func @"funct_partial_binding
# CHECK-SAME: <[[X:.*]][x]: !T, [[F:.*]][F]:
fn funct_partial_binding[x: T, F: fn[t: T, s: T] () -> None]():
    # CHECK: !lit.signature<<"u": !T, "v": !T>() -> !kgen.none> = <rebind(
    # CHECK-SAME: :!lit.signature<<"t": !T, "s": !T>() -> !kgen.none>
    # CHECK-SAME: bind_signature(:!lit.signature<<"t": !T, "s": !T>() -> !kgen.none> [[F]], ?, ?)

    alias G: fn[u: T, v: T] () -> None = F[s=_, t=_]
    # CHECK: !lit.signature<<"u": !T>() -> !kgen.none> = <rebind(
    # CHECK-SAME: :!lit.signature<<"s": !T>() -> !kgen.none>
    # CHECK-SAME: bind_signature(:!lit.signature<<"t": !T, "s": !T>() -> !kgen.none> [[F]], [[X]], ?))>
    alias H: fn[u: T] () -> None = F[x, _]
