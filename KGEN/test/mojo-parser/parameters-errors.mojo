# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Input parameters
##===----------------------------------------------------------------------===##

struct ParametricOnInt[a: Int]:
  pass

fn Rec2[
  a: ParametricOnInt[b], # expected-error {{use of unknown declaration 'b'}}
  b: ParametricOnInt[a]]():
  pass

# expected-note @+1 {{'Thing' declared here}}
struct Thing[a: Int, b: Int]:
  pass

fn GoodUseOfThing(a: Thing[4, 5]):
  pass

fn MissingThingMetaParams(a: Thing):  # expected-error {{'Thing' expects 2 input parameters but 0 were provided}}
  pass

# expected-error-re @below {{cannot apply more parameters to an already parameterized type 'Thing[{{.*}}1{{.*}}, {{.*}}2{{.*}}]'}}
fn MultipleThingMetaparams(a: Thing[1,2][1]):
  pass

fn OneMissingThingMetaParam(a: Thing[1]):  # expected-error {{'Thing' expects 2 input parameters but 1 was provided}}
  pass

# expected-error @+1 {{'Thing' parameter #1 has 'Int' type, but value has type 'FloatLiteral'}}
fn WeirdMetaParams(a: Thing[1, 1.5]):
  pass

struct Parameterized[p1: Int]:
  # expected-error @below {{invalid redefinition of 'p2'}}
  # expected-note @below {{previous definition here}}
  fn b[p2: Int, p2: Int, p3: Int](self): # Cannot shadow parameter names.
    pass

  fn __init__(inout self):
    pass

  # expected-note @+2 {{'method' declared here}}
  # expected-note @+1 {{function declared here}}
  fn method[B: Int](self, other: Parameterized[p1+B]):
    pass


# Test that we support partially bound parameters and diagnose incorrect uses
# of parameters.
fn testTestParamStruct(a: Parameterized[4]):
  a.method[7](Parameterized[11]())
  # expected-error-re @below {{invalid call to 'method': method argument #0 cannot be converted from 'Parameterized[{{.*}}12{{.*}}]' to 'Parameterized[{{.*}}11{{.*}}]'}}
  a.method[7](Parameterized[12]())
  a.method[2](Parameterized[6]())
  a.method[2, 7] # expected-error {{'method' expects 2 input parameters but 3 were provided}}


struct MySIMD[size: Int, type: __mlir_type.`!kgen.dtype`]:
  # expected-note @below {{function declared here}}
  fn __add__(self, rhs: MySIMD[size, type]):
    pass

fn testSIMD(a: MySIMD[1, __mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`],
            b: MySIMD[2, __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`]):
  var x = a+a
  var y = b+b
  var z = b+a  # expected-error-re {{invalid call to '__add__': right side cannot be converted from 'MySIMD[{{.*}}1{{.*}}, f64]' to 'MySIMD[{{.*}}2{{.*}}, si32]'}}

fn badReboundType[type: __mlir_type.`!kgen.dtype`,
                  val: __mlir_type[`!pop.scalar<`, type, `>`]]():
  pass

fn badCallReboundType[val: __mlir_type.`!pop.scalar<f32>`]():
  # expected-error @+1 {{cannot pass 'scalar<f32>' value, parameter expected 'scalar<f64>'}}
  badReboundType[__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`, val]()

fn partialBindSignature[callable: fn[a: Int, b: Int]() -> None, a: __mlir_type.index]():
  # expected-error @below {{parametric callable expected 2 parameters}}
  return callable[a]

# expected-note @+1 {{function declared here}}
def generic_fn[a: DType, b: Int](c : Int):
  pass

def call_generic[dt: DType]():
  generic_fn[dt, 1, 42](57) # expected-error {{invalid call to 'generic_fn': callee expects 2 input parameters but 3 were provided}}
  generic_fn[1, dt](57) # expected-error {{cannot pass 'IntLiteral' value, parameter expected 'DType'}}

fn meta_param_then_param_redef[
      dt: __mlir_type.index # expected-note {{previous definition here}}
    ](dt: __mlir_type.index):    # expected-error {{invalid redefinition of 'dt'}}
  pass

# expected-note @below {{previous definition here}}
# expected-error @below {{invalid redefinition of 'x'}}
def param_redef(x: __mlir_type.index, x: __mlir_type.index):
  pass

# expected-error @+1 {{non-default parameter follows default parameter}}
fn default_after_non_default[a: Int = 7, b: Int]():
    pass

##===----------------------------------------------------------------------===##
# Variadic Parameters
##===----------------------------------------------------------------------===##

# The keyword argument flags work in parameter lists.
fn funnyArgs[a: Int, /, b: Int, *,
             c: Int](): # expected-error {{keyword-only arguments not supported yet}}
    pass

# expected-error @+1 {{result parameters may not be variadic}}
fn variadicResultParams[() -> *b: Int]():
  pass

fn variadicIntParams[*a: Int]():
  pass

fn callVariadic():
  variadicIntParams() # OK
  variadicIntParams[1]() # OK
  variadicIntParams[1, 2]() # OK

  variadicIntParams[1.0]() # expected-error {{cannot pass 'FloatLiteral' value, parameter expected 'Int'}}



##===----------------------------------------------------------------------===##
# Result parameters
##===----------------------------------------------------------------------===##

# expected-error @+1 {{struct declarations do not support result parameters}}
struct ResultParams[a: Int -> b: Int, c: Float32]:
  pass

# expected-note @+1 {{function declared here}}
fn hasResultParam[a: Int -> b: Int]():
  param_return[a]

# expected-note @+1 {{function declared here}}
fn hasInputParam[a: Int]():
  pass

fn useResultParams():
  # expected-error @+1 {{invalid call to 'hasResultParam': callee expects 1 result parameter but 0 were provided}}
  hasResultParam[1]()

  alias b: Int
  hasResultParam[1 -> b]()  # expected-note {{previously defined here}}

  # expected-error @+1 {{'b' alias was defined by another result}}
  hasResultParam[1 -> b]()

  alias c: Int
  alias d: Int
  # expected-error @+1 {{invalid call to 'hasResultParam': callee expects 1 result parameter but 2 were provided}}
  hasResultParam[1 -> c, d]()

  # expected-error @+1 {{unable to find forward-declared alias named 'e'}}
  hasResultParam[1 -> e]()

  alias f: __mlir_type.f32 # expected-note {{alias forward declared here}}
  # expected-error @+1 {{result parameter returns type 'Int' but forward declaration is of type 'f32'}}
  hasResultParam[1 -> f]()

  var varNotAlias : __mlir_type.f32 # expected-note {{'varNotAlias' declared here}}
  # expected-error @+1 {{'varNotAlias' is not a forward declared alias}}
  hasResultParam[1 -> varNotAlias]()

  alias g: Int
  # expected-error @+1 {{calls with result parameter bindings must be called directly}}
  _ = hasResultParam[1 -> g]

  # expected-error-re @+1 {{cannot use parameterized function of type 'fn[Int]() -> None' without binding all its parameters}}
  var float1 = hasInputParam

  # expected-error @+1 {{invalid call to 'hasInputParam': callee expects 1 input parameter but 0 were provided}}
  hasInputParam()

  # Issue #6856: Cannot bind parameter results is parameter expressions
  alias h: Int
  # expected-error @+1 {{cannot call 'hasResultParam' in parameter expression because it has a parameter result}}
  alias x = hasResultParam[1 -> h]()

fn incorrectParameterReturnType[()-> a: Int]():
  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int' in return parameter}}
  param_return[4.0]

# expected-note @below {{function declared here}}
fn take_simd8(x: SIMD[DType.float32, 8]):
    pass

fn add_param_arg[x: Int](y: Int) -> Int:
    return x + y

fn pass_simd():
    # expected-error @below {{cannot be converted from 'SIMD[f32, add_param_arg[$builtin::$int::Int][8](8)]' to 'SIMD[f32, 8]'}}
    take_simd8(SIMD[DType.float32, add_param_arg[8](8)]())
    alias bar = add_param_arg
    # expected-error @below {{cannot be converted from 'SIMD[f32, bar[8](8)]' to 'SIMD[f32, 8]'}}
    take_simd8(SIMD[DType.float32, bar[8](8)]())

# expected-error @+1 {{unexpected default value for result parameter}}
fn default_result_param[a: Int -> b: Int = 7]():
    param_return[5]

##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##


alias x : Int # expected-error {{parameter results may only be declared in a function}}

fn testAliases(variable: Int):
  # expected-error @+1 {{declaration must have either a type or an initializer}}
  alias MissingInit

  # expected-error @+1 {{cannot use a dynamic value in alias initializer}}
  alias NotConstant = variable*2

  # TODO(Issue #5975): This isn't getting resolved before the end of body.
  # xpected-error @+1 {{alias 'NotInitialized' was never defined by a result parameter}}
  alias NotInitialized : __mlir_type.index

struct BadAliasStruct:
  alias x: Int # expected-error {{parameter results may only be declared in a function}}


fn testConversionQoI():
  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int' in alias initializer}}
  alias intVal : Int = 1.2


@always_inline("nodebug")
fn crash1_callee(a: __mlir_type.index, rhs: __mlir_type.index) -> __mlir_type.index:
  return __mlir_op.`index.add`(a, rhs)

fn crash1_caller[p: __mlir_type.index](a: __mlir_type.index):
  alias y = crash1_callee(a, p) # expected-error {{cannot use a dynamic value in alias initializer}}

@value
@register_passable
struct StructWithParam[n: Int]:
    alias Alias = StructWithParam[1]()

alias accessStructWithParam = StructWithParam.Alias # expected-error {{incorrect number of struct parameters}}


##===----------------------------------------------------------------------===##
# Default struct parameters
##===----------------------------------------------------------------------===##

# expected-error @below {{non-default parameter follows default parameter}}
struct DefaultParams[a: Int, b: Int = 7, msg: StringLiteral]:
    pass
