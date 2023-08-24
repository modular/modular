# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -verify-diagnostics | kgen-opt -verify-parameters | FileCheck %s


##===----------------------------------------------------------------------===##
# Input parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.struct.decl @OurSIMD
# CHECK-SAMEL <[[SIMDSIZE:.*]]: {{.*}}@"$int"::@Int, [[SIMDDT:.*]]: @"$builtin"::@"$dtype"::@DType>
# CHECK-SAME: attributes {registerPassable = 1 : i8} {
@register_passable
struct OurSIMD[size: Int, dt: DType]:
  fn __copyinit__(self) -> Self: pass


@register_passable
struct StructWithIntParam[size: Int]:
  pass

# CHECK-LABEL: lit.func @"paramArith{{.*}}"<{{.*}}x: {{.*}}@Int>() -> !lit.none
fn paramArith[x: Int]():
  # CHECK: kgen.call @"$builtin"::@"$constrained"::@"constrained[{{.*}}$bool::Bool]()"<{{.*}}apply({{.*}}__eq__{{.*}}, {{.*}}x, {{.*}}-99{{.*}})>()
  constrained[x == -100 + 1]()

fn take_3index(a: Int, b: Int, c: Int) -> Int:
  return a

# CHECK-LABEL: lit.func @"fancy_signature
# CHECK-SAME: <[[DT:.*]]: @"$builtin"::@"$dtype"::@DType, [[SIZE:.*]]: @"$builtin"::@"$int"::@Int>
# CHECK-SAME: (%x: !kgen.declref<@"$parameters"::@OurSIMD<[[SIMDSIZE:.*]]: @"$builtin"::@"$int"::@Int = [[SIZE]], [[SIMDDT:.*]]: @"$builtin"::@"$dtype"::@DType = [[DT]]>> borrow,
# CHECK-SAME: %exp: !kgen.declref<@"$parameters"::@OurSIMD<[[SIMDSIZE]]: @"$builtin"::@"$int"::@Int = [[SIZE]], [[SIMDDT]]: @"$builtin"::@"$dtype"::@DType = [[DT]]>> borrow) -> !kgen.declref<@"$builtin"::@"$int"::@Int>
fn fancy_signature[dt: DType, size: Int]
  (x: OurSIMD[size, dt], exp: (OurSIMD)[size, dt]) -> Int:

  # CHECK: %local = lit.varlet.decl "local", var = true
  # CHECK: %[[TMP1:.*]] = kgen.param.constant{{.*}}Int = <[[SIZE]]>
  # CHECK: %[[TMP2:.*]] = kgen.param.constant{{.*}}Int = <[[SIZE]]>
  # CHECK: %[[TMP3:.*]] = kgen.param.constant{{.*}}Int = <[[SIZE]]>
  # CHECK: %[[RES:.*]] = kgen.call @"$parameters"::@"take_3index{{.*}}(%[[TMP1]], %[[TMP2]], %[[TMP3]])
  # CHECK: pop.store %[[RES]], %local : !pop.pointer<{{.*}}@"$int"::@Int>
  var local = take_3index(size, size, size)

  # CHECK: %[[TMP:.*]] = kgen.call {{.*}}__add__
  # CHECK: lit.return %[[TMP]]
  return size+42


fn generic_fn[a: DType, b: Int, c: __mlir_type.`!kgen.mlirtype`](d : Int):
  pass

# CHECK: lit.func @"call_generic{{.*}}"<[[DT:.*]]: @"$builtin"::@"$dtype"::@DType>()
fn call_generic[dt: DType]():
  # CHECK: %[[C57:.*]] = {{.*}}constant{{.*}} 57
  # CHECK: kgen.call @"$parameters"::@"generic_fn{{.*}}"<:@"$builtin"::@"$dtype"::@DType [[DT]], :{{.*}}@"$int"::@Int {{.*}}42{{.*}}, :type @"$builtin"::@"$dtype"::@DType>(%[[C57]])
  generic_fn[dt, 42, DType](57)

  # CHECK: %[[TMP:.*]] = {{.*}}constant{{.*}} 57
  # CHECK: kgen.call @"$parameters"::@"generic_fn{{.*}}"<:@"$builtin"::@"$dtype"::@DType [[DT]], :{{.*}}@"$int"::@Int #lit.struct<{value = 13}>, :type @"$parameters"::@OurSIMD<[[SIMDSIZE:.*]]: {{.*}}@"$int"::@Int = #lit.struct<{value = 4}>, {{.*}}dt: @"$builtin"::@"$dtype"::@DType = [[DT]]>>(%2) : (!kgen.declref<{{.*}}@"$int"::@Int> borrow) -> !lit.none
  generic_fn[dt, 13, OurSIMD[4, dt]](57)

# CHECK-LABEL: lit.struct.decl @TestParamStruct<
# CHECK-SAME: [[A:.*]]: {{.*}}@Int>
@register_passable
struct TestParamStruct[A: Int]:

  fn __copyinit__(self) -> Self:
    return Self{}

  fn __init__() -> TestParamStruct[A]:
    return TestParamStruct[A]{}

  # CHECK: lit.func @"method{{.*}}<[[B:.*]]: {{.*}}@Int>(%self: !kgen.declref<{{.*}}TestParamStruct<[[A]]: {{.*}}@Int = [[A]]>> borrow,
  # CHECK-SAME: %other: !kgen.declref<@"$parameters"::@TestParamStruct<[[A]]: {{.*}}@Int = apply({{.*}}__add__{{.*}}, [[A]], [[B]])>> borrow)
  fn method[B: Int](self: TestParamStruct[A], other: TestParamStruct[A+B]):
    pass

  # CHECK-LABEL: lit.func @"aliases{{.*}}%x: !kgen.declref<@"$parameters"::@TestParamStruct<
  fn aliases(self, x: TestParamStruct[TestParamStruct[A].TypeLevelAlias]):
    # CHECK: lit.alias.decl [[B:.*]]: {{.*}}@Int = <apply({{.*}}__add__{{.*}}, apply({{.*}}__mul__{{.*}}, [[A]], [[A]]), {{.*}}1{{.*}})>
    alias B = A*A+1
    # CHECK: lit.alias.decl [[C:.*]]: {{.*}}@Int = <apply({{.*}}__mul__{{.*}}, [[B]], [[A]])>
    alias C = B*A
    # CHECK: lit.alias.decl [[D:.*]]: {{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = {{.*}}1{{.*}}> = <apply(:<>() ownedresult -> {{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = {{.*}}1{{.*}}>> {{.*}}__init__()"<:{{.*}}@Int {{.*}}1
    alias D = TestParamStruct[1]()
    # CHECK: %temp = lit.varlet.decl {{.*}} : <{{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = [[C]]>>
    var temp: TestParamStruct[C]

    # CHECK: lit.alias.decl {{.*}}intVal: {{.*}}@"$int"::@Int = <#lit.struct<{value = 42}>>
    alias intVal : Int = 42

    # CHECK: %temp2 = lit.varlet.decl {{.*}} : <{{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = apply({{.*}}__mul__{{.*}}, [[A]], [[A]])
    var temp2: TestParamStruct[TestParamStruct[A].TypeLevelAlias]

  # CHECK: lit.alias.decl {{.*}}TypeLevelAlias: {{.*}}@Int = <apply({{.*}}__mul__{{.*}}, [[A]], [[A]])
  alias TypeLevelAlias = A*A

# Test that we support partially bound parameters.
fn testTestParamStruct(a: TestParamStruct[4]):
  # CHECK: %arg11 = lit.varlet.decl {{.*}} : <{{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = {{.*}}11
  # CHECK: %0 = kgen.call {{.*}}@TestParamStruct::@"__init__{{.*}}<:{{.*}}@Int {{.*}}11{{.*}}>() : () ownedresult -> !kgen.declref<{{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = {{.*}}11
  # CHECK: pop.store %0, %arg11 : !pop.pointer<{{.*}}@TestParamStruct<[[A]]: {{.*}}@Int = {{.*}}11
  var arg11 = TestParamStruct[11]()

  # CHECK: %1 = pop.load %arg11
  # CHECK: kgen.call {{.*}}@TestParamStruct::@"method{{.*}}<{{.*}}4{{.*}}7{{.*}}>(%a, %2)
  a.method[7](arg11)

# CHECK-LABEL: lit.func @"testSIMD(
fn testSIMD(a: SIMD[DType.float64, 1],
            b: SIMD[DType.int32, 1],
            inout reff: SIMD[DType.int32, 1]):
  # CHECK: %field1 = lit.varlet.decl {{.*}} : <scalar<f64>>
  var field1 = a.value
  # CHECK: %field2 = lit.varlet.decl {{.*}} : <scalar<si32>>
  var field2 = reff.value

  # Test calls to methods and operators on parameterized type.
  _ = a.fma(a, a)
  _ = b.fma(b, b)
  # CHECK: kgen.call @"$builtin"::@"$simd"::@SIMD::@"__add__({{.*}}<:{{.*}} dtype = f64{{.*}}, {{.*}} = 1{{.*}}>(%a, %a)
  var x = a+a
  # CHECK: kgen.call @"$builtin"::@"$simd"::@SIMD::@"__add__({{.*}}<:{{.*}} dtype = si32{{.*}}, {{.*}} = 1{{.*}}>(%b, %b)
  var y = b+b

# Show that forward references of parameter names can be correctly resolved.
#
# CHECK-LABEL: lit.func @"paramResolution[
# CHECK-SAME: $int::Int,
# CHECK-SAME: $parameters::StructWithIntParam[*(0,0)],
# CHECK-SAME: $int::Int,
# CHECK-SAME: $parameters::StructWithIntParam[*(0,2)]
# CHECK-SAME: ]()"<
# CHECK-SAME: [[SIZE1:.*]]: {{.*}}@"$int"::@Int, {{.*}}a: @"$parameters"::@StructWithIntParam<[[S:.*]]: {{.*}}@"$int"::@Int = [[SIZE1]]>,
# CHECK-SAME: [[SIZE2:.*]]: {{.*}}@"$int"::@Int, {{.*}}b: @"$parameters"::@StructWithIntParam<[[S]]: {{.*}}@"$int"::@Int = [[SIZE2]]>>()
fn paramResolution[size1: Int, a: StructWithIntParam[size1],
                   size2: Int, b: StructWithIntParam[size2]]():
  pass

# Show that we can implicitly convert from 42's literal type to Int.
# CHECK-LABEL: lit.func @"implConversion
# CHECK: <{{.*}}a: @"$parameters"::@StructWithIntParam<{{.*}}size: {{.*}}@"$int"::@Int = #lit.struct<{{.*}}42}>>
fn implConversion[a: StructWithIntParam[42]]():
  pass

# CHECK-LABEL: lit.struct.decl @Pair<
# CHECK-SAME: [[DT:.*]]: @"$builtin"::@"$dtype"::@DType>
@register_passable
struct Pair[dt: DType]:
 # CHECK: lit.struct.field a : !kgen.declref<@"$parameters"::@OurSIMD<[[SIMDSIZE:.*]]: {{.*}}@"$int"::@Int = {{.*}}42{{.*}}, [[SIMDDT:.*]]: @"$builtin"::@"$dtype"::@DType = [[DT]]>>
 # CHECK: lit.struct.field b : !kgen.declref<{{.*}}@"$int"::@Int>
  var a : OurSIMD[42, dt]
  var b : Int

  # CHECK: lit.func @"__init__{{.*}} -> !kgen.declref<@"$parameters"::@Pair<[[DT]]: @"$builtin"::@"$dtype"::@DType = [[DT]]>> attributes {{.*}} isStatic
  fn __init__(a: OurSIMD[42, dt]) -> Pair[dt]:
    # CHECK: [[TMP:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%a)
    # CHECK: %1 = kgen.param.constant: {{.*}}@"$int"::@Int {{.*}} 4
    # CHECK: %2 = lit.struct.create(a=%0, b=%1) : (!kgen.declref<@"$parameters"::@OurSIMD<[[SIMDSIZE]]: {{.*}}@"$int"::@Int = #lit.struct<{value = 42}>, [[SIMDDT]]: @"$builtin"::@"$dtype"::@DType = [[DT]]>>, !kgen.declref<{{.*}}@"$int"::@Int>) -> !kgen.declref<@"$parameters"::@Pair<[[DT]]: @"$builtin"::@"$dtype"::@DType = [[DT]]>>
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
  # CHECK: [[TMP1:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK:  = lit.struct.create(a=[[TMP1]], b=%b)
  return Pair[DType.float32]{a: a, b: b}

# CHECK-LABEL: lit.struct.decl @TypeParameter
# CHECK-SAME: <[[TYPE:.*]]: type>
struct TypeParameter[type: __mlir_type.`!kgen.mlirtype`]:
  # CHECK: @"bar($parameters::TypeParameter{{.*}})"(%self: {{.*}} borrow_in_mem, %val: !kgen.paramref<[[TYPE]]> borrow)
  fn bar(self, val: type):
    pass

# Test that parameter decls can refine subsequent ones in the same param list.
# CHECK-LABEL: lit.struct.decl @ParamSubst
# CHECK-SAME: <[[TYPE:.*]]: type, [[SH:.*]]: variadic<[[TYPE]]>>
struct ParamSubst[
    type: AnyType,
    shape: __mlir_type[`!kgen.variadic<`, type,`>`],
  ]: pass

# CHECK-LABEL: lit.func @"testParamSubst
fn testParamSubst():
  # CHECK: %xx = lit.varlet.decl {{.*}} : <@"$parameters"::@ParamSubst<[[TYPE]]: type = index, [[SH]]: variadic<index> = [1, 2]>>
  var xx : ParamSubst[__mlir_type.index, __mlir_attr.`#kgen.variadic<1, 2> : !kgen.variadic<index>`]


# Test parameter substitution.
# CHECK-LABEL: lit.func @"fnToCall{{.*}}"
# CHECK-SAME: <[[SIZE:.*]], [[ARR:.*]]: array<[[SIZE]], f32>>()
fn fnToCall[size: __mlir_type.index, array: __mlir_type[`!pop.array<`, size, `, f32>`]]():
  pass

# CHECK: lit.func @"fnWithCall{{.*}}"
# CHECK-SAME: <[[ARR:.*]]: array<10, f32>
fn fnWithCall[array: __mlir_type[`!pop.array<10, f32>`]]():
   # CHECK: kgen.call @"$parameters"::@"fnToCall{{.*}}"<10, :array<10, f32> [[ARR]]>()
   fnToCall[(10).value, array]()

# CHECK-LABEL: lit.func @"meta_str{{.*}}"<{{.*}}type: {{.*}}@StringLiteral>() -> !lit.none
fn meta_str[type: StringLiteral]():
  pass

# CHECK-LABEL: lit.func @"str_input_param()"() -> !lit.none
fn str_input_param():
  # CHECK: %0 = kgen.call @"$parameters"::@"meta_str{{.*}}"<:{{.*}}@StringLiteral {{.*}}"123"{{.*}}>() : () -> !lit.none
  meta_str["123"]()

@value
@register_passable
struct TwoParams[a: Int, b: Int]:
    pass

# CHECK-LABEL: lit.func @"signature_capture
# CHECK-SAME: {{.*}}a: {{.*}}Int, {{.*}}f: <{{.*}}Int>() ownedresult ->
# CHECK-SAME: {{.*}}TwoParams<{{.*}}a: {{.*}}Int = {{.*}}a, {{.*}}b: {{.*}}Int = *(0,0)>
fn signature_capture[a: Int, f: fn[b: Int]() -> TwoParams[a, b]]():
    _ = f[2]()

##===----------------------------------------------------------------------===##
# Memory-primary parameters
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
    # CHECK: %[[PARAM_VALUE:.*]] = kgen.param.materialize: {{.*}}MemoryType = <{{.*}}paramValue>
    # CHECK: pop.store %[[PARAM_VALUE]], %dynamicLet
    let dynamicLet = paramValue

    alias nonMovable = NonMovableMemoryType(42)
    # CHECK: %dynamicVar = lit.varlet.decl
    # CHECK: %[[NON_MOVABLE:.*]] = kgen.param.materialize: {{.*}}NonMovableMemoryType
    # CHECK: pop.store %[[NON_MOVABLE]], %dynamicVar
    var dynamicVar = nonMovable

    # CHECK: copy: {{.*}}MemoryType = <apply_result_slot({{.*}}passMemoryValue{{.*}}, #M.store_to_mem<{{.*}}paramValue
    alias copy = passMemoryValue(paramValue)
    # CHECK: lit.varlet.decl
    # CHECK: %[[MVALUE:.*]] = lit.varlet.decl
    # CHECK: %[[PVALUE:.*]] = kgen.param.materialize: {{.*}}@MemoryType = <{{.*}}copy>
    # CHECK: pop.store %[[PVALUE]], %[[MVALUE]]
    # CHECK: call {{.*}}passMemoryValue{{.*}}(%{{.*}}, %[[MVALUE]])
    _ = passMemoryValue(copy)

    # CHECK: call {{.*}}memoryParam{{.*}}<:{{.*}}MemoryType apply_result_slot({{.*}}__init__{{.*}}value = 22
    memoryParam[MemoryType(22)]()

# CHECK-LABEL: lit.func @"memoryParam
# CHECK-SAME: <{{.*}}value: {{.*}}MemoryType>()
fn memoryParam[value: MemoryType]():
    pass

##===----------------------------------------------------------------------===##
# Result parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"no_result_param{{.*}}"<{{.*}}a: {{.*}}@Int>()
fn no_result_param[a: Int]():
  return

# CHECK-LABEL: lit.func @"idx_result_params[{{.*}}$int::Int]()"
# CHECK-SAME: <[[A:.*]]: {{.*}}@"$int"::@Int -> {{.*}}b: {{.*}}@"$int"::@Int, {{.*}}c: {{.*}}@"$int"::@Int>()
fn idx_result_params[a: Int -> b: Int, c: Int]() -> Int:
  # CHECK: lit.param_return<{{.*}}@Int [[A]], {{.*}}apply({{.*}}__add__{{.*}}, [[A]], {{.*}}1
  param_return[a, a+1]
  # CHECK: %0 = kgen.param.constant: {{.*}}@Int = <[[A]]>
  # CHECK: %1 = kgen.param.constant: {{.*}}@Int = {{.*}}2
  # CHECK: %2 = kgen.call {{.*}}__add__{{.*}}(%0, %1)
  # CHECK-NEXT: lit.return %2
  # CHECK-NEXT: kgen.param.result_bind<{{.*}}?, {{.*}}?>
  return a+2

# CHECK-LABEL: lit.func @"parametric_result_params{{.*}}"
# CHECK-SAME: <[[T:.*]]: type, [[INPUT:.*]]: !kgen.paramref<[[T]]> ->
fn parametric_result_params[T: AnyType, input: T -> out: T]():
    # CHECK: lit.param_return<:!kgen.paramref<[[T]]> [[INPUT]]>
    # CHECK: kgen.param.result_bind<:!kgen.paramref<[[T]]> ?>
    param_return[input]

# CHECK-LABEL: lit.func @"just_result_params{{.*}}"<() -> {{.*}}a>()
fn just_result_params[() -> a: __mlir_type.index]():
  # CHECK: lit.param_return<42>
  param_return[(42).value]


# CHECK-LABEL: lit.func @"result_param_ref()"
fn result_param_ref():
    # CHECK: unbound_ref: <[] -> index>() -> !lit.none = <{{.*}}@"just_result_params()">
    alias unbound_ref = just_result_params
    # CHECK: bound_ref: <[] -> {{.*}}Int, {{.*}}Int>() -> {{.*}}Int> = <{{.*}}idx_result_params{{.*}}<:{{.*}}Int #lit.struct<{value = 1}>>>
    alias bound_ref = idx_result_params[1]


# CHECK-LABEL: lit.func @"search3{{.*}}"<{{.*}}a: {{.*}}@"$int"::@Int, {{.*}}b: {{.*}}@"$int"::@Int, {{.*}}c: {{.*}}@"$int"::@Int -> {{.*}}d: {{.*}}@"$int"::@Int>()
fn search3[a: Int, b: Int, c: Int -> d: Int]():
   param_return[a]

# CHECK-LABEL: lit.func @"useResultParams
fn useResultParams(i: Int):
  # Call function with input parameter, no result parameters.
  # CHECK: kgen.call @"$parameters"::@"no_result_param{{.*}}"<{{.*}}42{{.*}}>()
  no_result_param[42]()

  # CHECK: lit.alias.fwd.decl "[[XYZ:.*]]" : {{.*}}@Int
  alias xyz: Int

  # Normal result and multi parameter results.  This forward references xyz
  # CHECK: [[TMP:%.*]] = kgen.call {{.*}}@"idx_result_params{{.*}}"<:{{.*}}@Int apply({{.*}}__mul__{{.*}}, [[XYZ]], {{.*}}2
  # CHECK-NEXT: kgen.call {{.*}}@"$int"::@Int::@"__mul__({{.*}}$int::Int,{{.*}}$int::Int)"([[TMP]], %i)
  alias a: Int
  alias b: Int
  _ = idx_result_params[xyz*2 -> a, b]() * i

  # CHECK: kgen.call {{.*}}@"search3{{.*}}"<{{.*}}1{{.*}}2{{.*}}3{{.*}} -> [[XYZ]]: {{.*}}@Int>()
  search3[1,2,3 -> xyz]()

  # CHECK: kgen.call {{.*}}@"no_result_param{{.*}}"<{{.*}}apply({{.*}}__add__{{.*}}, [[XYZ]], {{.*}}1
  no_result_param[xyz+1]()

  # Function call with only a result parameter.
  # CHECK: lit.alias.fwd.decl "[[C:.*]]" : index
  alias c : __mlir_type.index
  # CHECK: kgen.call @"$parameters"::@"just_result_params{{.*}}"<[] -> [[C]]>()
  just_result_params[() -> c]()

# CHECK-LABEL: lit.func @"testParamInIf
fn testParamInIf(c: Bool):
    # CHECK: hlcf.if
    if c:
        # CHECK-NEXT: alias.fwd.decl "[[X:.*]]"
        alias x: __mlir_type.index
        # CHECK-NEXT: call {{.*}}<[] -> [[X]]>
        just_result_params[() -> x]()

# Issue #6904: Parameter results don't get implicit conversions
# CHECK-LABEL: lit.func @"testResultParamConversion
fn testResultParamConversion[() -> b: Int](a: Int):
  # CHECK: lit.param_return<:{{.*}}@"$int"::@Int #lit.struct<{{.*}} 4}
  param_return[4]

# CHECK-LABEL: lit.func @"testResultParamThrowing()"<() -> {{.*}}b:
fn testResultParamThrowing[() -> b: Int]() raises:
  # CHECK: lit.param_return<:{{.*}}@"$int"::@Int #lit.struct<{{.*}} 1}
  param_return[1]
  # CHECK: lit.return %{{.*}} : !pop.variant<@{{.*}}::@Error, !lit.none>
  raise Error()

# CHECK-LABEL: lit.func @"testMultipleParamReturn[{{.*}}$bool::Bool]()"<{{.*}}a: {{.*}} -> {{.*}}b:
fn testMultipleParamReturn[a: Bool -> b: Int]():
    # CHECK: kgen.param.if
    @parameter
    if a:
        # CHECK: lit.param_return<{{.*}} 1}
        param_return[1]
    # CHECK: else
    else:
        # CHECK: lit.param_return<{{.*}} 2}
        param_return[2]

##===----------------------------------------------------------------------===##
# First-class functions as parameters.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"takeCallable{{.*}}"
# CHECK-SAME: <[[CALLABLE:.*]]: <>(index borrow) -> index>(%a: index borrow) -> index
fn takeCallable[
     callable: __mlir_type[`!kgen.signature<(index borrow) -> index>`]
   ](a: __mlir_type.index) -> __mlir_type.index:
  # CHECK-NEXT: %0 = kgen.call_param[<>(index borrow) -> index: [[CALLABLE]]](%a)
  # CHECK-NEXT: lit.return %0
  return callable(a)

fn takeAndReturnIndex(x: __mlir_type.index) -> __mlir_type.index:
  return x

# CHECK-LABEL: lit.func @"takeAndReturnIndex
fn passFunction(a: __mlir_type.index) -> __mlir_type.index:
  # CHECK: %0 = kgen.call @"$parameters"::@"takeCallable{{.*}}<:<>(index borrow) -> index {{.*}}@"takeAndReturnIndex{{.*}}">(%a)
  return takeCallable[takeAndReturnIndex](a)

##===--------------------Test function with parameters---------------------===##

# CHECK-LABEL: lit.func @"callableWithParam{{.*}}"<{{.*}}type: dtype>() -> !lit.none
fn callableWithParam[type: __mlir_type.`!kgen.dtype`]():
  pass

# CHECK-LABEL: lit.func @"takeCallable2
fn takeCallable2[
      func: fn[dt: __mlir_type.`!kgen.dtype`]() -> None
  ]():
      pass

# CHECK-LABEL: lit.func @"passFunctionParam2
fn passFunctionParam2():
  # CHECK: kgen.call @"$parameters"::@"takeCallable2{{.*}}"<
  # CHECK-SAME: :<dtype>() -> !lit.none @"$parameters"::@"callableWithParam{{.*}}">()
  takeCallable2[callableWithParam]()

# CHECK-LABEL: lit.func @"my_constrained{{.*}}()"
# CHECK-SAME: <[[COND:.*]]: {{.*}}@"$bool"::@Bool, [[MESSAGE:.*]]: {{.*}}@StringLiteral>
fn my_constrained[cond: Bool, message: StringLiteral]():
    # CHECK: kgen.param.assert <apply({{.*}}__mlir_i1__{{.*}}, [[COND]])>, #lit.struct.extract<{{.*}}[[MESSAGE]], "value">
    __mlir_op.`kgen.param.assert`[cond:cond.__mlir_i1__(), message:message.value]()
    return


# CHECK-LABEL: lit.func @"pass_str_param
fn pass_str_param():
    # CHECK: kgen.call {{.+}}my_constrained{{.*}}"<{{.*}}true{{.*}}, :{{.*}}@StringLiteral {{.*}}"foo"{{.*}}>()
    my_constrained[1==1, "foo"]()

##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl {{.*}}boolDtype: dtype = <bool>
alias boolDtype = __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
# CHECK: lit.alias.decl {{.*}}FOURTY_TWO: {{.*}}@Int = <{{.*}}42
alias FOURTY_TWO = 42

# CHECK-LABEL: lit.struct.decl @A
# CHECK-SAME: <[[V:.*]]: {{.*}}@Int>
struct A[v: Int]:
  # CHECK: lit.alias.decl {{.*}}member: {{.*}}@Int = <apply({{.*}}__add__{{.*}}, [[V]], {{.*}}42
  alias member = v + FOURTY_TWO

# CHECK-LABEL: lit.func @"testUseOfAliases
fn testUseOfAliases(a: Bool):
  # This type checks.
  _ = SIMD[DType(boolDtype), 4].splat(a)
  # CHECK: lit.alias.decl {{.*}}y: {{.*}}@Int = <{{.*}}44
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

  alias ui8 = MyDType((1).value)
  alias float32 = MyDType((2).value)
  alias float64 = MyDType((3).value)

  # CHECK: lit.alias.decl {{.*}}ui16: @"$parameters"::@MyDType = <#lit.struct<{state = 7}>>
  alias ui16 = MyDType{state: (7).value}

struct MyVector[size: Int, dtype: MyDType]:
  pass

fn testMyDType[dt: MyDType](a: MyVector[4, MyDType.float32],
                            b: MyVector[4, dt]):

   constrained[dt == MyDType.float64]()

# Issue #6828: Unqualified name lookup into structs doesn't work
# CHECK-LABEL: lit.struct.decl @UnqualAliasLookup
# CHECK-SAME: <[[PARAM:.*]]: {{.*}}@Int>
struct UnqualAliasLookup[param: Int]:
  # CHECK: lit.alias.decl {{.*}}member: {{.*}}@Int = <apply({{.*}}__add__{{.*}}, [[PARAM]], {{.*}}1{{.*}})>
  alias member = param+1
  fn get(self) -> Int:
    # CHECK: %0 = kgen.param.constant: {{.*}}@Int = <apply({{.*}}__add__{{.*}}, [[PARAM]], {{.*}}1{{.*}})>
    return Self.member

##===----------------------------------------------------------------------===##
# Variadic parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"fnWithVariadics{{.*}}()"<{{.*}}b: variadic<{{.*}}@"$int"::@Int>>
fn fnWithVariadics[*b: Int]():
  pass

# CHECK-LABEL: lit.struct.decl @StructWithVariadics
# CHECK-SAME: <[[B:.*]]: variadic<{{.*}}@"$int"::@Int>>
struct StructWithVariadics[*b: Int]:
    fn __init__(inout self, i: Int):
        pass

# CHECK-LABEL: lit.func @"useParamVariadics
fn useParamVariadics():
  # CHECK-NEXT: kgen.call @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<{{.*}}@"$int"::@Int> []>()
  fnWithVariadics()

  # CHECK: kgen.call @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<{{.*}}@"$int"::@Int> [#lit.struct<{value = 1}>]>()
  fnWithVariadics[1]()
  # CHECK: kgen.call @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<{{.*}}@"$int"::@Int> [#lit.struct<{value = 1}>, #lit.struct<{value = 2}>]>()
  fnWithVariadics[1, 2]()

  # This keeps the parameters unbound, allowing them to be used with different length..
  # CHECK-NEXT: lit.alias.decl {{.*}}fnAlias: <variadic<{{.*}}@"$int"::@Int>>() param_vararg -> !lit.none = <@"$parameters"::@"fnWithVariadics{{.*}}">
  alias fnAlias = fnWithVariadics

  # Use of an unbound thing in a DRValue context binds an empty variadic list.
  # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure [<>() param_vararg -> !lit.none: @"$parameters"::@"fnWithVariadics{{.*}}"<:variadic<{{.*}}@"$int"::@Int> []>]()
  # CHECK-NEXT:  %fnLet = lit.letreg.decl "fnLet" = [[TMP]] : !kgen.signature<() param_vararg -> !lit.none>
  let fnLet = fnWithVariadics

  # CHECK-NEXT: %a = lit.varlet.decl {{.*}} <@"{{.*}}::@StructWithVariadics<[[B]]: variadic<{{.*}}@"$int"::@Int> = []>>
  var a: StructWithVariadics
  # CHECK-NEXT: %b = lit.varlet.decl {{.*}} : <@{{.*}}::@StructWithVariadics<[[B]]: variadic<{{.*}}@"$int"::@Int> = [#lit.struct<{value = 1}>]>>
  var b: StructWithVariadics[1]
  # CHECK-NEXT: %c = lit.varlet.decl {{.*}} : <@{{.*}}::@StructWithVariadics<[[B]]: variadic<{{.*}}@"$int"::@Int> = [#lit.struct<{value = 1}>, #lit.struct<{value = 2}>]>>
  var c: StructWithVariadics[1, 2]

  # TODO(16040): fix symbol name mangling to erase parameter name 'b'
  # CHECK: kgen.call {{.*}}@StructWithVariadics::@"__init__(${{.*}}::StructWithVariadics[b]=&,{{.*}}$int::Int)"<:variadic<{{.*}}@"$int"::@Int> [#lit.struct<{value = 1}>]>
  var d = StructWithVariadics[1](2)
  # CHECK: kgen.call {{.*}}@StructWithVariadics::@"__init__(${{.*}}::StructWithVariadics[b]=&,{{.*}}$int::Int)"<:variadic<{{.*}}@"$int"::@Int> []>
  var e = StructWithVariadics(3)


# CHECK-LABEL: lit.func @"variadic_parameter{{.*}}"<{{.*}}elems: variadic<index>>
fn variadic_parameter[elems: __mlir_type.`!kgen.variadic<index>`]() -> Int:
    return 3

fn dependent_variadic_parameter[
    type: __mlir_type.`!kgen.mlirtype`, *values: type
](): pass

# CHECK-LABEL: lit.func @"pass_variadic{{.*}}"
# CHECK-SAME: <[[ELEMS:.*]]: variadic<index>>
fn pass_variadic[elems: __mlir_type.`!kgen.variadic<index>`]():
    # CHECK-NEXT: kgen.call @"$parameters"::@"variadic_parameter{{.*}}"<:variadic<index> [[ELEMS]]>
    _ = variadic_parameter[elems]()
    # CHECK: kgen.call @"$parameters"::@"dependent_variadic_parameter{{.*}}"<:type {{.*}}@"$int"::@Int, :variadic<{{.*}}@"$int"::@Int>
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
fn callee2[T: __mlir_type.`!kgen.mlirtype`](v: T): pass
fn callee3[size: __mlir_type.index, type: __mlir_type.`!kgen.dtype`]
   (v:  __mlir_type[`!pop.simd<`, size, `, `, type, `>`]): pass
fn callee4[T: __mlir_type.`!kgen.mlirtype`]
   (v:  __mlir_type[`!pop.pointer<`, T, `>`]): pass

# CHECK-LABEL: lit.func @"testParamInference{{.*}}"<
# CHECK-SAME: [[SIZE:.*]]: {{.*}}@"$int"::@Int>(
fn testParamInference[size: Int](a: StaticVec[4], b: StaticVec[size],
                                 b2: StaticVec[size*2],
                                 c: __mlir_type.`!pop.simd<17, f32>`,
                                 d: __mlir_type.`!pop.pointer<f32>`):
  # CHECK-NEXT: kgen.call @{{.*}}callee1{{.*}}<{{.*}}4{{.*}}>(%a)
  callee1(a)
  # CHECK-NEXT: kgen.call @{{.*}}callee1{{.*}}<{{.*}}@Int [[SIZE]]>(%b)
  callee1(b)
  # CHECK-NEXT: kgen.call @{{.*}}callee1{{.*}}<{{.*}}@Int apply({{.*}}__mul__{{.*}}, [[SIZE]], {{.*}}2{{.*}})>(%b2)
  callee1(b2)
  # CHECK-NEXT: kgen.call @{{.*}}callee2{{.*}}<:type @"$parameters"::@StaticVec<{{.*}}size: {{.*}}@Int = [[SIZE]]>>(%b)
  callee2(b)
  # CHECK-NEXT: kgen.call @{{.*}}callee3{{.*}}<17, :dtype f32>(%c)
  callee3(c)
  # CHECK-NEXT: kgen.call @{{.*}}callee4{{.*}}<:type f32>(%d)
  callee4(d)

# CHECK-LABEL: lit.struct.decl @Abstraction
# CHECK-SAMEL <[[A:.*]]: {{.*}}@"$int"::@Int>
@register_passable
struct Abstraction[a: Int]:
  alias val = a.value

  fn __init__() -> Self:
      return Self{}

  fn __copyinit__(self) -> Self:
      return self

  @staticmethod
  fn push[b: Int]() -> Abstraction[a + b]:
      return Abstraction[a + b]()

  @staticmethod
  fn pull[b: Int](value: Abstraction[a + b]):
      return

# CHECK-LABEL: lit.func @"testDependentType{{.*}}"<
# CHECK-SAME: [[RANK:.*]]: {{.*}}@"$int"::@Int, {{.*}}shape: array<#lit.struct.extract<:{{.*}}@"$int"::@Int [[RANK]], "value">
fn testDependentType[
    rank: Int,
    shape: __mlir_type[`!pop.array<`, rank.value, `, index>`],
]():
    pass

# CHECK-LABEL: lit.func @"testParameterEvaluator()"
fn testParameterEvaluator():
  # CHECK-NEXT: lit.alias.decl {{.*}}x = <1>
  alias x = Abstraction[1].val
  # CHECK-NEXT: %0 = kgen.call @"$parameters"::@Abstraction::@"push{{.*}}"<:{{.*}} = 1{{.*}}, :{{.*}} = 2{{.*}}>()
  # CHECK-NEXT: %1 = kgen.rebind %0 : {{.*}} to {{.*}}@Abstraction<[[A:.*]]: {{.*}} = 3}
  # CHECK-NEXT: %y = lit.letreg.decl "y" = %1
  let y : Abstraction[3] = Abstraction[1].push[2]()
  # CHECK-NEXT: %2 = kgen.rebind %y : {{.*}}@Abstraction<[[A]]: {{.*}} = 3}
  # CHECK-NEXT: kgen.call {{.*}}@Abstraction::@"pull{{.*}}"<{{.*}}>(%2)
  Abstraction[1].pull[2](y)
  # CHECK-NEXT: kgen.call {{.*}}@"testDependentType{{.*}}"<:{{.*}} = 1{{.*}}, :array<1, index>
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
    # CHECK: %[[VALUE_PTR:.*]] = lit.struct.gep %lvalue[value]
    # CHECK-NEXT: kgen.rebind %[[VALUE_PTR]] {{.*}} to
    # CHECK-SAME: !pop.pointer<{{.*}}@Abstraction<[[A:.*]]: {{.*}} = {{.*}} 2}>>>
    takeAbstraction2(lvalue.value)
    let rvalue = AnotherAbstraction[1]()
    # CHECK: %[[VALUE:.*]] = lit.struct.extract %rvalue[value]
    # CHECK-NEXT: kgen.rebind %[[VALUE]] {{.*}} to {{.*}}@Abstraction<[[A]]: {{.*}} = {{.*}} 2}>>
    takeAbstraction2(rvalue.value)


fn tail_types[T: AnyType, *U: AnyType](a: T, *b: *U):
    pass

# CHECK-LABEL: lit.func @"call_with_tail_types()"
fn call_with_tail_types():
    # CHECK: call {{.*}}tail_types{{.*}}<:type {{.*}}@Int, :variadic<type> []>
    tail_types(1)
    # CHECK: call {{.*}}tail_types{{.*}}<:type {{.*}}@Int, :variadic<type> [{{.*}}FloatLiteral]>
    tail_types(1, 1.2)
