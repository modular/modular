# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s -I %S/../mojo-examples/ -verify-diagnostics | kgen-opt -verify-parameters | FileCheck %s

from prolog import assert_param, assert_param, SIMD
from DType import DType

##===----------------------------------------------------------------------===##
# Input parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.struct.decl @OurSIMD<size: @"$Int"::@Int, dt: @"$DType"::@DType>
# CHECK-SAME: attributes {registerPassable = 1 : i8} {
@register_passable
struct OurSIMD[size: Int, dt: DType]:
  fn __copyinit__(self) -> Self: pass


@register_passable
struct StructWithIntParam[size: Int]:
  pass

# CHECK-LABEL: lit.func @"paramArith{{.*}}"<x>() -> !lit.none
fn paramArith[x: __mlir_type.index]():
  # CHECK: kgen.call @"$Assert"::@"assert_param()"{{.*}}eq(x, -99)
  assert_param[x == (-100).__as_mlir_index()+(1).__as_mlir_index()]()

fn take_3index(a: Int, b: Int, c: Int) -> Int:
  return a

# CHECK-LABEL: lit.func @"fancy_signature
# CHECK-SAME: <dt: @"$DType"::@DType, size: @"$Int"::@Int>
# CHECK-SAME: (%x: !kgen.declref<@"$parameters"::@OurSIMD<size: @"$Int"::@Int = size, dt: @"$DType"::@DType = dt>> borrow,
# CHECK-SAME:  %exp: !kgen.declref<@"$parameters"::@OurSIMD<size: @"$Int"::@Int = size, dt: @"$DType"::@DType = dt>> borrow) -> !kgen.declref<@"$Int"::@Int>
fn fancy_signature[dt: DType, size: Int]
  (x: OurSIMD[size, dt], exp: (OurSIMD)[size, dt]) -> Int:

  # CHECK: %[[TMP1:.*]] = kgen.param.constant{{.*}}Int = <size>
  # CHECK: %[[TMP2:.*]] = kgen.param.constant{{.*}}Int = <size>
  # CHECK: %[[TMP3:.*]] = kgen.param.constant{{.*}}Int = <size>
  # CHECK: %[[RES:.*]] = kgen.call @"$parameters"::@"take_3index{{.*}}(%[[TMP1]], %[[TMP2]], %[[TMP3]])
  # CHECK: %local = lit.varlet.decl "local", var = true
  # CHECK: pop.store %[[RES]], %local : !pop.pointer<@"$Int"::@Int>
  var local = take_3index(size, size, size)

  # CHECK: %[[TMP:.*]] = kgen.call {{.*}}__add__
  # CHECK: lit.return %[[TMP]]
  return size+42


fn generic_fn[a: DType, b: Int, c: __mlir_type.`!kgen.mlirtype`](d : Int):
  pass

# CHECK: lit.func @"call_generic()"<dt: @"$DType"::@DType>()
fn call_generic[dt: DType]():
  # CHECK: %[[C57:.*]] = {{.*}}constant{{.*}} 57
  # CHECK: kgen.call @"$parameters"::@"generic_fn{{.*}}<:@"$DType"::@DType dt, :@"$Int"::@Int {{.*}}42{{.*}}, :type !kgen.declref<@"$DType"::@DType>>(%[[C57]])
  generic_fn[dt, 42, DType](57)

  # CHECK: %[[TMP:.*]] = {{.*}}constant{{.*}} 57
  # CHECK: kgen.call @"$parameters"::@"generic_fn($Int::Int)"<:@"$DType"::@DType dt, :@"$Int"::@Int #lit.struct<{value: scalar<index> = 13}>, :type !kgen.declref<@"$parameters"::@OurSIMD<size: @"$Int"::@Int = #lit.struct<{value: scalar<index> = 4}>, dt: @"$DType"::@DType = dt>>>(%2) : (!kgen.declref<@"$Int"::@Int> borrow) -> !lit.none
  generic_fn[dt, 13, OurSIMD[4, dt]](57)

# CHECK-LABEL: lit.struct.decl @TestParamStruct<A>
@register_passable
struct TestParamStruct[A: __mlir_type.index]:

  fn __copyinit__(self) -> Self:
    return Self{}

  fn __init__() -> TestParamStruct[A]:
    return TestParamStruct[A]{}

  # CHECK: lit.func @"method{{.*}}<B>(%self: !kgen.declref<{{.*}}TestParamStruct<A = A>> borrow,
  # CHECK-SAME: %other: !kgen.declref<@"$parameters"::@TestParamStruct<A = add(A, B)>> borrow)
  fn method[B: __mlir_type.index](self: TestParamStruct[A], other: TestParamStruct[A+B]):
    pass

  # CHECK-LABEL: lit.func @"aliases{{.*}}%x: !kgen.declref<@"$parameters"::@TestParamStruct<
  fn aliases(self, x: TestParamStruct[TestParamStruct[A].TypeLevelAlias]):
    # CHECK: kgen.param.declare B = <add(mul(A, A), 1)>
    alias B = A*A+(1).__as_mlir_index()
    # CHECK: kgen.param.declare C = <mul(A, B)>
    alias C = B*A
    # CHECK: kgen.param.declare D: @"$parameters"::@TestParamStruct<A = 1> = <apply(:<>() ownedresult -> !kgen.declref<@"$parameters"::@TestParamStruct<A = 1>> @"$parameters"::@TestParamStruct::@"__init__()"<1>)>
    alias D = TestParamStruct[(1).__as_mlir_index()]()
    # CHECK: %temp = lit.varlet.decl {{.*}} : <@"$parameters"::@TestParamStruct<A = C>>
    var temp: TestParamStruct[C]

    # CHECK: kgen.param.declare intVal: @"$Int"::@Int = <#lit.struct<{value: scalar<index> = 42}>>
    alias intVal : Int = 42

    # CHECK:  %temp2 = lit.varlet.decl {{.*}} : <@"$parameters"::@TestParamStruct<A = mul(A, A)>>
    var temp2: TestParamStruct[TestParamStruct[A].TypeLevelAlias]

  # CHECK: kgen.param.declare TypeLevelAlias = <mul(A, A)>
  alias TypeLevelAlias = A*A

# Test that we support partially bound parameters.
fn testTestParamStruct(a: TestParamStruct[(4).__as_mlir_index()]):
  # CHECK: %0 = kgen.call @"$parameters"::@TestParamStruct::@"__init__{{.*}}<11>() : () ownedresult -> !kgen.declref<@"$parameters"::@TestParamStruct<A = 11>>
  # CHECK: %arg11 = lit.varlet.decl {{.*}} : <@"$parameters"::@TestParamStruct<A = 11>>
  # CHECK: pop.store %0, %arg11 : !pop.pointer<@"$parameters"::@TestParamStruct<A = 11>>
  var arg11 = TestParamStruct[(11).__as_mlir_index()]()

  # CHECK: %1 = pop.load %arg11
  # CHECK: %2 = kgen.call @"$parameters"::@TestParamStruct::@"method{{.*}}<4, 7>(%a, %1)
  a.method[(7).__as_mlir_index()](arg11)

# CHECK-LABEL: lit.func @"testSIMD(
fn testSIMD(a: SIMD[DType.f64, 1],
            b: SIMD[DType.si32, 1],
            ref&: SIMD[DType.si32, 1]):
  # CHECK: %field1 = lit.varlet.decl {{.*}} : <scalar<f64>>
  var field1 = a.value
  # CHECK: %field2 = lit.varlet.decl {{.*}} : <scalar<si32>>
  var field2 = ref.value

  # Test calls to methods and operators on parameterized type.
  _ = a.fma(a, a)
  _ = b.fma(b, b)
  # CHECK: kgen.call @"$SIMD"::@SIMD::@"__add__{{.*}}<:{{.*}} dtype = f64{{.*}}, {{.*}} = 1{{.*}}>(%a, %a)
  var x = a+a
  # CHECK: kgen.call @"$SIMD"::@SIMD::@"__add__{{.*}}<:{{.*}} dtype = si32{{.*}}, {{.*}} = 1{{.*}}>(%b, %b)
  var y = b+b

# Show that forward references of parameter names can be correctly resolved.
#
# CHECK-LABEL: lit.func @"paramResolution()"<
# CHECK-SAME: size1: @"$Int"::@Int,
# CHECK-SAME: a: @"$parameters"::@StructWithIntParam<size: @"$Int"::@Int = size1>,
# CHECK-SAME: size2: @"$Int"::@Int,
# CHECK-SAME: b: @"$parameters"::@StructWithIntParam<size: @"$Int"::@Int = size2>>()
fn paramResolution[size1: Int, a: StructWithIntParam[size1],
                   size2: Int, b: StructWithIntParam[size2]]():
  pass

# Show that we can implicitly convert from 42's literal type to Int.
# CHECK-LABEL: lit.func @"implConversion
# CHECK: <a: @"$parameters"::@StructWithIntParam<size: @"$Int"::@Int = #lit.struct<{{.*}}42}>>
fn implConversion[a: StructWithIntParam[42]]():
  pass

# CHECK-LABEL: lit.struct.decl @Pair<dt: @"$DType"::@DType>
@register_passable
struct Pair[dt: DType]:
 # CHECK: lit.struct.field a : !kgen.declref<@"$parameters"::@OurSIMD<size{{.*}} = {{.*}}42{{.*}}, dt: @"$DType"::@DType = dt>>
 # CHECK: lit.struct.field b : !kgen.declref<@"$Int"::@Int>
  var a : OurSIMD[42, dt]
  var b : Int

  # CHECK: lit.func @"__init__{{.*}} -> !kgen.declref<@"$parameters"::@Pair<dt: @"$DType"::@DType = dt>> attributes {{.*}} isStatic
  fn __init__(a: OurSIMD[42, dt]) -> Pair[dt]:
    # CHECK: [[TMP:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%a)
    # CHECK: %1 = kgen.param.constant: @"$Int"::@Int {{.*}} 4
    # CHECK: %2 = lit.struct.create(a=%0, b=%1) : (!kgen.declref<@"$parameters"::@OurSIMD<size: @"$Int"::@Int = #lit.struct<{value: scalar<index> = 42}>, dt: @"$DType"::@DType = dt>>, !kgen.declref<@"$Int"::@Int>) -> !kgen.declref<@"$parameters"::@Pair<dt: @"$DType"::@DType = dt>>
    return Pair[dt]{a: a, b: 4}
  # CHECK: }

  fn __copyinit__(self) -> Self: pass

# CHECK: }

# CHECK: useParameterizedField
fn useParameterizedField[x: Pair[DType.f32]]():
  # CHECK: kgen.param.declare y:
  alias y : OurSIMD[42, DType.f32] = x.a


# CHECK-LABEL: lit.func @"makePair
fn makePair(a: OurSIMD[42, DType.f32], b: Int) -> Pair[DType.f32]:
  # CHECK: [[TMP1:%.*]] = kgen.call {{.*}}__copyinit__{{.*}}(%a)
  # CHECK:  = lit.struct.create(a=[[TMP1]], b=%b)
  return Pair[DType.f32]{a: a, b: b}

# CHECK-LABEL: lit.struct.decl @TypeParameter
struct TypeParameter[type: __mlir_type.`!kgen.mlirtype`]:
  # CHECK: @"bar($parameters::TypeParameter{{.*}})"(%self: {{.*}} borrow_in_mem, %val: !kgen.paramref<type> borrow)
  fn bar(self, val: type):
    pass

# Test that parameter decls can refine subsequent ones in the same param list.
# CHECK-LABEL: lit.struct.decl @ParamSubst<type: type, shape: variadic<type>>
struct ParamSubst[
    type: AnyType,
    shape: __mlir_type[`!kgen.variadic<`, type,`>`],
  ]: pass

# CHECK-LABEL: lit.func @"testParamSubst
fn testParamSubst():
  # CHECK: %xx = lit.varlet.decl {{.*}} : <@"$parameters"::@ParamSubst<type: type = index, shape: variadic<index> = [1, 2]>>
  var xx : ParamSubst[__mlir_type.index, __mlir_attr.`#kgen.variadic<1, 2> : !kgen.variadic<index>`]


# Test parameter substitution.
# CHECK-LABEL: lit.func @"fnToCall()"<size, array: array<size, f32>>()
fn fnToCall[size: __mlir_type.index, array: __mlir_type[`!pop.array<`, size, `, f32>`]]():
  pass

# CHECK: lit.func @"fnWithCall
fn fnWithCall[array: __mlir_type[`!pop.array<10, f32>`]]():
   # CHECK:  kgen.call @"$parameters"::@"fnToCall()"<10, :array<10, f32> array>()
   fnToCall[(10).__as_mlir_index(), array]()

# CHECK-LABEL: lit.func @"meta_str()"<type: {{.*}}@StringLiteral>() -> !lit.none
fn meta_str[type: StringLiteral]():
  pass

# CHECK-LABEL: lit.func @"str_input_param()"() -> !lit.none
fn str_input_param():
  # CHECK: %0 = kgen.call @"$parameters"::@"meta_str()"<:{{.*}}@StringLiteral {{.*}}"123"{{.*}}>() : () -> !lit.none
  meta_str["123"]()

##===----------------------------------------------------------------------===##
# Result parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"no_result_param()"<a>()
fn no_result_param[a: __mlir_type.index]():
  return

# CHECK-LABEL: lit.func @"idx_result_params()"<a -> b, c>()
fn idx_result_params[a: __mlir_type.index -> b: __mlir_type.index, c: __mlir_type.index]() -> Int:
  # CHECK-NEXT: lit.param_return<a, add(a, 1)>
  param_return[a,a+(1).__as_mlir_index()]
  # CHECK: %0 = kgen.param.constant = <add(a, 2)>
  # CHECK: %1 = kgen.call{{.*}}__init__{{.*}}(%0)
  # CHECK-NEXT: lit.return %1
  # CHECK-NEXT: kgen.param.result_bind<?, ?>
  return a+(2).__as_mlir_index()

# CHECK-LABEL: lit.func @"parametric_result_params()"
fn parametric_result_params[T: AnyType, input: T -> out: T]():
    # CHECK: lit.param_return<:!kgen.paramref<T> input>
    # CHECK: kgen.param.result_bind<:!kgen.paramref<T> ?>
    param_return[input]

# CHECK-LABEL: lit.func @"just_result_params()"<() -> a>()
fn just_result_params[() -> a: __mlir_type.index]():
  # CHECK: lit.param_return<42>
  param_return[(42).__as_mlir_index()]


# A bit grotty, but effective way to provide a kgen.param.fork over three
# values.  This should be made variadic and generic and moved to the standard
# library at some point.

# CHECK-LABEL: lit.func @"search3()"<a, b, c -> d>()
fn search3[a: __mlir_type.index, b: __mlir_type.index, c: __mlir_type.index
              -> d: __mlir_type.index]():

   # Grotty but effective way to pull a param decl out and return it.
   # CHECK-NEXT: kgen.param.fork result_hidden = <[a, b, c]>
   __mlir_op.`kgen.param.fork`[
      paramDecl: __mlir_attr.`#kgen<param.decl result_hidden : index>`,
      values: __mlir_attr[`#kgen.variadic<`, a, `, `, b, `, `, c, `> : !kgen.variadic<index>`]
   ]()
   # CHECK: kgen.param.declare result = <result_hidden>
   alias result = __mlir_attr.`#kgen.param.decl.ref<"result_hidden"> : index`
   # CHECK: lit.param_return<result>
   param_return[result]

# CHECK-LABEL: lit.func @"useResultParams
fn useResultParams(i: Int):
  # Call function with input parameter, no result parameters.
  # CHECK: kgen.call @"$parameters"::@"no_result_param()"<42>()
  no_result_param[(42).__as_mlir_index()]()

  # CHECK: lit.alias.fwd.decl "xyz" : index
  alias xyz : __mlir_type.index

  # Normal result and multi parameter results.  This forward references xyz
  # CHECK: [[TMP:%.*]] = kgen.call @"$parameters"::@"idx_result_params()"<mul(xyz, 2) -> a, b>() : () -> !kgen.declref<@"$Int"::@Int>
  # CHECK-NEXT: kgen.call @"$Int"::@Int::@"__mul__($Int::Int,$Int::Int)"([[TMP]], %i)
  alias a : __mlir_type.index
  alias b : __mlir_type.index
  _ = idx_result_params[xyz*(2).__as_mlir_index() -> a, b]() * i

  # CHECK: kgen.call @"$parameters"::@"search3()"<1, 2, 3 -> xyz>()
  search3[(1).__as_mlir_index(),(2).__as_mlir_index(),(3).__as_mlir_index()-> xyz]()

  # CHECK: kgen.call @"$parameters"::@"no_result_param()"<add(xyz, 1)>()
  no_result_param[xyz+(1).__as_mlir_index()]()

  # Function call with only a result parameter.
  alias c : __mlir_type.index
  # CHECK: kgen.call @"$parameters"::@"just_result_params()"<[] -> c>()
  just_result_params[() -> c]()

# Issue #6904: Parameter results don't get implicit conversions
# CHECK: lit.func @"testResultParamConversion
fn testResultParamConversion[() -> b: Int](a: Int):
  # CHECK: lit.param_return<:@"$Int"::@Int #lit.struct<{{.*}} 4}
  param_return[4]

# lit.func @"testResultParamThrowing()"<() -> b:
fn testResultParamThrowing[() -> b: Int]() raises:
  # CHECK: lit.param_return<:@"$Int"::@Int #lit.struct<{{.*}} 1}
  param_return[1]
  # CHECK: lit.return %1 : !pop.variant<@{{.*}}::@Error, !lit.none>
  raise Error()

# lit.func @"testMultipleParamReturn()"<a: {{.*}} -> b:
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

# CHECK-LABEL: lit.func @"takeCallable{{.*}}"<callable: <>(index borrow) -> index>(%a: index borrow) -> index
fn takeCallable[
     callable: __mlir_type[`!kgen.signature<(index borrow) -> index>`]
   ](a: __mlir_type.index) -> __mlir_type.index:
  # CHECK-NEXT: %0 = kgen.call_param[<>(index borrow) -> index: callable](%a)
  # CHECK-NEXT: lit.return %0
  return callable(a)

fn takeAndReturnIndex(x: __mlir_type.index) -> __mlir_type.index:
  return x

# CHECK-LABEL: lit.func @"takeAndReturnIndex
fn passFunction(a: __mlir_type.index) -> __mlir_type.index:
  # CHECK: %0 = kgen.call @"$parameters"::@"takeCallable{{.*}}<:<>(index borrow) -> index {{.*}}@"takeAndReturnIndex{{.*}}">(%a)
  return takeCallable[takeAndReturnIndex](a)

##===--------------------Test function with parameters---------------------===##

# CHECK-LABEL: lit.func @"callableWithParam()"<type: dtype>() -> !lit.none
fn callableWithParam[type: __mlir_type.`!kgen.dtype`]():
  pass

# CHECK-LABEL: lit.func @"takeCallable2
fn takeCallable2[
      func: fn[dt: __mlir_type.`!kgen.dtype`]() -> None
  ]():
      pass

# CHECK-LABEL: lit.func @"passFunctionParam2
fn passFunctionParam2():
  #CHECK: kgen.call @"$parameters"::@"takeCallable2()"
  #CHECK-SAME: <:<dtype>() -> !lit.none {{.*}}@"callableWithParam()">() : () -> !lit.none
  takeCallable2[callableWithParam]()

# CHECK-LABEL: lit.func @"my_assert_param()"<cond: i1, message: {{.*}}@StringLiteral>() -> !lit.none
fn my_assert_param[cond: __mlir_type.i1, message: StringLiteral]():
    #CHECK: kgen.param.assert <cond>, #lit.struct.extract<{{.*}}message, "value">
    __mlir_op.`kgen.param.assert`[cond:cond, message:message.value]()
    return


# CHECK-LABEL: lit.func @"pass_str_param
fn pass_str_param():
    # CHECK: kgen.call {{.+}}my_assert_param()"<:i1 1, :{{.*}}@StringLiteral {{.*}}"foo"{{.*}}>() : () -> !lit.none
    my_assert_param[(1).__as_mlir_index()==(1).__as_mlir_index(), "foo"]()

##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##

# CHECK: kgen.param.declare boolDtype: dtype = <bool>
alias boolDtype = __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
# CHECK: kgen.param.declare FOURTY_TWO = <42>
alias FOURTY_TWO = (42).__as_mlir_index()

# CHECK-LABEL: lit.struct.decl @A<v>
struct A[v: __mlir_type.index]:
  # CHECK: kgen.param.declare member = <add(v, 42)>
  alias member = v + FOURTY_TWO

# CHECK-LABEL: lit.func @"testUseOfAliases
fn testUseOfAliases(a: Bool):
  # This type checks.
  _ = SIMD[DType(boolDtype), 4].splat(a.value)
  # CHECK: kgen.param.declare y = <44>
  alias y = A[(2).__as_mlir_index()].member

@register_passable
struct MyDType:
  var state : __mlir_type.index

  fn __copyinit__(self) -> Self:
    return Self{state: self.state}

  fn __init__(value: __mlir_type.index) -> MyDType:
     return MyDType{state: value}

  fn __eq__(self, rhs: MyDType) -> Bool:
     return True  # TODO: buggy impl :-)

  alias ui8 = MyDType((1).__as_mlir_index())
  alias f32 = MyDType((2).__as_mlir_index())
  alias f64 = MyDType((3).__as_mlir_index())

  # CHECK: kgen.param.declare *"ui16": @"$parameters"::@MyDType = <#lit.struct<{state = 7}>>
  alias ui16 = MyDType{state: (7).__as_mlir_index()}

struct MyVector[size: Int, dtype: MyDType]:
  pass

fn testMyDType[dt: MyDType](a: MyVector[4, MyDType.f32],
                            b: MyVector[4, dt]):

   assert_param[dt == MyDType.f64]()

# Issue #6828: Unqualified name lookup into structs doesn't work
# CHECK-LABEL: lit.struct.decl @UnqualAliasLookup<param>
struct UnqualAliasLookup[param: __mlir_type.index]:
  # CHECK: kgen.param.declare member = <add(param, 1)>
  alias member = param+(1).__as_mlir_index()
  fn get(self) -> __mlir_type.index:
    # CHECK: %0 = kgen.param.constant = <add(param, 1)>
    return member

##===----------------------------------------------------------------------===##
# Variadic parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.func @"fnWithVariadics
# CHECK-SAME: <b: variadic<@"$Int"::@Int>>
fn fnWithVariadics[*b: Int]():
  pass

# CHECK-LABEL: lit.struct.decl @StructWithVariadics
# CHECK-SAME: <b: variadic<@"$Int"::@Int>>
struct StructWithVariadics[*b: Int]:
    fn __init__(inout self, i: Int):
        pass

# CHECK-LABEL: lit.func @"useParamVariadics
fn useParamVariadics():
  # CHECK-NEXT: kgen.call @"$parameters"::@"fnWithVariadics()"<:variadic<@"$Int"::@Int> []>()
  fnWithVariadics()

  # CHECK: kgen.call @"$parameters"::@"fnWithVariadics()"<:variadic<@"$Int"::@Int> [#lit.struct<{value: scalar<index> = 1}>]>()
  fnWithVariadics[1]()
  # CHECK: kgen.call @"$parameters"::@"fnWithVariadics()"<:variadic<@"$Int"::@Int> [#lit.struct<{value: scalar<index> = 1}>, #lit.struct<{value: scalar<index> = 2}>]>()
  fnWithVariadics[1, 2]()

  # This keeps the parameters unbound, allowing them to be used with different length..
  # CHECK-NEXT: kgen.param.declare fnAlias: <variadic<@"$Int"::@Int>>() param_vararg -> !lit.none = <@"$parameters"::@"fnWithVariadics()">
  alias fnAlias = fnWithVariadics

  # Use of an unbound thing in a DRValue context binds an empty variadic list.
  # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure [<>() param_vararg -> !lit.none: @"$parameters"::@"fnWithVariadics()"<:variadic<@"$Int"::@Int> []>]()
  # CHECK-NEXT:  %fnLet = lit.letreg.decl "fnLet" = [[TMP]] : !kgen.signature<() param_vararg -> !lit.none>
  let fnLet = fnWithVariadics

  # CHECK-NEXT: %a = lit.varlet.decl {{.*}} <@"{{.*}}::@StructWithVariadics<b: variadic<@"$Int"::@Int> = []>>
  var a: StructWithVariadics
  # CHECK-NEXT: %b = lit.varlet.decl {{.*}} : <@{{.*}}::@StructWithVariadics<b: variadic<@"$Int"::@Int> = [#lit.struct<{value: scalar<index> = 1}>]>>
  var b: StructWithVariadics[1]
  # CHECK-NEXT: %c = lit.varlet.decl {{.*}} : <@{{.*}}::@StructWithVariadics<b: variadic<@"$Int"::@Int> = [#lit.struct<{value: scalar<index> = 1}>, #lit.struct<{value: scalar<index> = 2}>]>>
  var c: StructWithVariadics[1, 2]

  # CHECK: kgen.call {{.*}}@StructWithVariadics::@"__init__(${{.*}}::StructWithVariadics[b]=&,$Int::Int)"<:variadic<@"$Int"::@Int> [#lit.struct<{value: scalar<index> = 1}>]>
  var d = StructWithVariadics[1](2)
  # CHECK: kgen.call {{.*}}@StructWithVariadics::@"__init__(${{.*}}::StructWithVariadics[b]=&,$Int::Int)"<:variadic<@"$Int"::@Int> []>
  var e = StructWithVariadics(3)


# CHECK-LABEL: lit.func @"variadic_parameter()"<elems: variadic<index>>
fn variadic_parameter[elems: __mlir_type.`!kgen.variadic<index>`]() -> Int:
    return 3

fn dependent_variadic_parameter[
    type: __mlir_type.`!kgen.mlirtype`, *values: type
](): pass

# CHECK-LABEL: lit.func @"pass_variadic()"<elems: variadic<index>>
fn pass_variadic[elems: __mlir_type.`!kgen.variadic<index>`]():
    # CHECK-NEXT: kgen.call @"$parameters"::@"variadic_parameter()"<:variadic<index> elems>
    _ = variadic_parameter[elems]()
    # CHECK: kgen.call @"$parameters"::@"dependent_variadic_parameter()"<:type !kgen.declref<@"$Int"::@Int>, :variadic<@"$Int"::@Int>
    _ = dependent_variadic_parameter[Int, 1, 2]()


##===----------------------------------------------------------------------===##
# Parameter Inference
##===----------------------------------------------------------------------===##

@register_passable("trivial")
struct StaticVec[size: __mlir_type.index]:
  fn __init__[type: __mlir_type.`!kgen.dtype`](v: __mlir_type[`!pop.simd<`, size, `, `, type, `>`]) -> StaticVec[size]:
      return Self{}

  @staticmethod
  fn thing[type: __mlir_type.`!kgen.dtype`](v: __mlir_type[`!pop.simd<`, size, `, `, type, `>`]):
      return

fn callee1[size: __mlir_type.index](v: StaticVec[size]): pass
fn callee2[T: __mlir_type.`!kgen.mlirtype`](v: T): pass
fn callee3[size: __mlir_type.index, type: __mlir_type.`!kgen.dtype`]
   (v:  __mlir_type[`!pop.simd<`, size, `, `, type, `>`]): pass
fn callee4[T: __mlir_type.`!kgen.mlirtype`]
   (v:  __mlir_type[`!pop.pointer<`, T, `>`]): pass

# CHECK-LABEL: lit.func @"testParamInference
fn testParamInference[size: __mlir_type.index](a: StaticVec[(4).__as_mlir_index()], b: StaticVec[size],
                                   b2: StaticVec[size*(2).__as_mlir_index()],
                                   c: __mlir_type.`!pop.simd<17, f32>`,
                                   d: __mlir_type.`!pop.pointer<f32>`):
  # CHECK-NEXT: kgen.call @{{.*}}callee1{{.*}}<4>(%a)
  callee1(a)
  # CHECK-NEXT: kgen.call @{{.*}}callee1{{.*}}<size>(%b)
  callee1(b)
  # CHECK-NEXT: kgen.call @{{.*}}callee1{{.*}}<mul(size, 2)>(%b2)
  callee1(b2)
  # CHECK-NEXT: kgen.call @{{.*}}callee2{{.*}}<:type !kgen.declref<@"$parameters"::@StaticVec<size = size>>>(%b)
  callee2(b)
  # CHECK-NEXT: kgen.call @{{.*}}callee3{{.*}}<17, :dtype f32>(%c)
  callee3(c)
  # CHECK-NEXT: kgen.call @{{.*}}callee4{{.*}}<:type f32>(%d)
  callee4(d)

@register_passable
struct Abstraction[a: Int]:
  alias val = a.__as_mlir_index()

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

# CHECK-LABEL: lit.func @"testDependentType()"
# CHECK-SAME: shape: array<apply{{.*}}@Int::@"__as_mlir_index
fn testDependentType[
    rank: Int,
    shape: __mlir_type[`!pop.array<`, rank.__as_mlir_index(), `, index>`],
]():
    pass

# CHECK-LABEL: lit.func @"testParameterEvaluator()"
fn testParameterEvaluator():
  # CHECK-NEXT: declare x = <1>
  alias x = Abstraction[1].val
  # CHECK-NEXT: %0 = kgen.call @"$parameters"::@Abstraction::@"push()"<:{{.*}} scalar<index> = 1{{.*}}, :{{.*}} scalar<index> = 2{{.*}}>()
  # CHECK-NEXT: %1 = kgen.rebind %0 : {{.*}} to {{.*}}@Abstraction<a: {{.*}} scalar<index> = 3}
  # CHECK-NEXT: %y = lit.letreg.decl "y" = %1
  let y : Abstraction[3] = Abstraction[1].push[2]()
  # CHECK-NEXT: %2 = kgen.rebind %y : {{.*}}@Abstraction<a: {{.*}} scalar<index> = 3}
  # CHECK-NEXT: kgen.call {{.*}}@Abstraction::@"pull{{.*}}"<{{.*}}>(%2)
  Abstraction[1].pull[2](y)
  # CHECK-NEXT: kgen.call {{.*}}@"testDependentType()"<:{{.*}} = 1{{.*}}, :array<apply{{.*}}__as_mlir_index{{.*}} rebind(:array<1, index>
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
    # CHECK-SAME: !pop.pointer<{{.*}}@Abstraction<a: {{.*}} = {{.*}} 2}>>>
    takeAbstraction2(lvalue.value)
    let rvalue = AnotherAbstraction[1]()
    # CHECK: %[[VALUE:.*]] = lit.struct.extract %rvalue[value]
    # CHECK-NEXT: kgen.rebind %[[VALUE]] {{.*}} to {{.*}}@Abstraction<a: {{.*}} = {{.*}} 2}>>
    takeAbstraction2(rvalue.value)


fn tail_types[T: AnyType, *U: AnyType](a: T, *b: *U):
    pass

# CHECK-LABEL: lit.func @"call_with_tail_types()"
fn call_with_tail_types():
    # CHECK: call {{.*}}tail_types{{.*}}<:type {{.*}}@Int>, :variadic<!kgen.mlirtype> []>
    tail_types(1)
    # CHECK: call {{.*}}tail_types{{.*}}<:type {{.*}}@Int>, :variadic<!kgen.mlirtype> [{{.*}}FloatLiteral>]>
    tail_types(1, 1.2)
