# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

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

# expected-error @below {{Thing' expects 0 parameters, but 1 was specified}}
fn MultipleThingMetaparams(a: Thing[1,2][1]):
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
  a.method[2, 7] # expected-error {{'method' expects 2 parameters, but 3 were specified}}

  var partial_var_type: Thing[1] # expected-error {{missing required parameter 'b'}}


alias DType = __mlir_type.`!kgen.dtype`

# expected-note @below {{struct declared here}}
struct MySIMD[size: Int, type: DType]:
  # expected-note @below {{function declared here}}
  fn __add__(self, rhs: MySIMD[size, type]):
    pass

# expected-note @below {{function declared here}}
fn twoUses[dt1: DType, dt2: DType, size: Int](lhs: MySIMD[size, dt1], rhs: MySIMD[size, dt2]):
  pass

fn testSIMD(a: MySIMD[1, __mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`],
            b: MySIMD[2, __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`]):
  var x = a+a
  var y = b+b
  # expected-error @below {{invalid call to '__add__': could not deduce parameter 'size' of parent struct 'MySIMD'}}
  # expected-note @below {{failed to infer parameter #0, parameter inferred to two different values: '2' and '1'}}
  var z = b+a

  # expected-error @below {{invalid call to 'twoUses': could not deduce parameter 'size' of callee 'twoUses'}}
  # expected-note @below {{failed to infer parameter 'size', parameter inferred to two different values: '1' and '2'}}
  twoUses(a, b)

struct TwoParams[a: int, b: int]:
    fn __init__(inout self, other: TwoParams[`1`, `1`]):
        pass

# expected-note @below {{function declared here}}
fn infer_then_convert[a: int, b: int](lhs: TwoParams[a, b],
                                      rhs: TwoParams[a, b]):
    pass

fn left_to_right_implicit_conversion(lhs: TwoParams[`1`, `2`],
                                     rhs: TwoParams[`1`, `1`]):
    # This succeeds because 'a' and 'b' are inferred to '1' and '2', and 'rhs'
    # can implicitly convert from 'TwoParams[1, 1]' to 'TwoParams[1, 2]'.
    infer_then_convert(lhs, rhs)
    # This fails because 'a' and 'b' are inferred to '1' and '1', and 'lhs'
    # cannot implicit convert from 'TwoParams[1, 2]' to 'TwoParams[1, 1]'.
    # expected-error @below {{invalid call to 'infer_then_convert': could not deduce parameter 'a' of callee 'infer_then_convert'}}
    # expected-note @below {{failed to infer parameter 'b', parameter inferred to two different values: '1' and '2'}}
    infer_then_convert(rhs, lhs)

fn badReboundType[type: DType, val: __mlir_type[`!pop.scalar<`, type, `>`]]():
  pass

fn badCallReboundType[val: __mlir_type.`!pop.scalar<f32>`]():
  # expected-error @+1 {{cannot pass 'scalar<f32>' value, expected 'scalar<f64>' in call parameter}}
  badReboundType[__mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`, val]()

# expected-note @+1 {{function declared here}}
def generic_fn[a: FloatLiteral, b: Int](c : Int):
  pass

def call_generic[dt: FloatLiteral]():
  # expected-error @+1 {{invalid call to 'generic_fn': callee expects 2 parameters, but 3 were specified}}
  generic_fn[dt, 1, 42](57)

fn meta_param_then_param_redef[
      dt: __mlir_type.index # expected-note {{previous definition here}}
    ](dt: __mlir_type.index):    # expected-error {{invalid redefinition of 'dt'}}
  pass

# expected-note @below {{previous definition here}}
# expected-error @below {{invalid redefinition of 'x'}}
def param_redef(x: __mlir_type.index, x: __mlir_type.index):
  pass

# expected-error @+1 {{required positional parameter follows optional positional parameter}}
fn default_after_non_default[a: Int = 7, b: Int]():
    pass

##===----------------------------------------------------------------------===##
# Variadic Parameters
##===----------------------------------------------------------------------===##

# expected-error @+1 {{variadic keyword parameters not supported yet}}
fn variadic_kw_result_binding[**a: Int]():
    pass

fn variadic_kw_binding[*a: Int]():
    variadic_kw_result_binding[**a]() # expected-error {{keyword unpacking not supported yet}}

# expected-note @below {{function declared here}}
fn variadic_int_params[*a: Int]():
    pass

fn callVariadic():
  variadic_int_params[1.0]() # expected-error {{callee parameter #0 has 'Int' type, but value has type 'FloatLiteral'}}

##===----------------------------------------------------------------------===##
# Function Overloading on Parameters
##===----------------------------------------------------------------------===##

struct ConvertibleFromInt:
    fn __init__(inout self, value: Int):
        pass

struct AlsoConvertibleFromInt:
    fn __init__(inout self, value: Int):
        pass

struct NotConvertible:
    pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: parameter_overloading parameter #0 has 'ConvertibleFromInt' type, but value has type 'NotConvertible'}}
fn parameter_overloading[param: ConvertibleFromInt]():
    pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: parameter_overloading parameter #0 has 'AlsoConvertibleFromInt' type, but value has type 'NotConvertible'}}
fn parameter_overloading[param: AlsoConvertibleFromInt]():
    pass


# expected-note @below {{candidate declared here}}
fn arg_overloading_with_param[param: Int](): pass
# expected-note @below {{candidate declared here}}
fn arg_overloading_with_param[param: Int](a: Int): pass

fn form_reference_to_overloaded[value: NotConvertible]():
    # expected-error @below {{cannot form a reference to overloaded declaration of 'parameter_overloading', each candidate requires 1 implicit conversion, disambiguate with an explicit cast}}
    alias ambiguous = parameter_overloading[1]
    # expected-error @below {{cannot form a reference to overloaded declaration of 'parameter_overloading'}}
    alias none_valid = parameter_overloading[value]

    # MOCO-1024: Bad error message with missing ()'s on UnsafePointer.load
    # expected-error @below {{cannot form a reference to overloaded declaration of 'arg_overloading_with_param'}}
    # expected-note @below {{did you mean to call it?}}
    arg_overloading_with_param[1]


##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##


fn testAliases(variable: Int):
  # expected-error @+1 {{expected '=' in alias declaration}}
  alias MissingInit

  # expected-error @+1 {{cannot use a dynamic value in alias initializer}}
  alias NotConstant = variable+2


fn testConversionQoI():
  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int'}}
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

alias accessStructWithParam = StructWithParam.Alias # expected-error {{incorrect number of type parameters: expected 1 but got 0}}


##===----------------------------------------------------------------------===##
# Default struct parameters
##===----------------------------------------------------------------------===##

# expected-error @below {{required positional parameter follows optional positional parameter}}
struct DefaultParams[a: Int, b: Int = 7, msg: StringLiteral]: pass


@value
struct DefaultParams2[a: Int, b: Int = 7]: pass  # expected-note {{declared here}}

fn test_default_param_struct():
    # expected-error @+1 {{expects at most 2 parameters, but 3 were specified}}
    alias S = DefaultParams2[1, 3, 4]

fn missing_bound_param():
    # expected-error @below {{missing required parameter 'a'}}
    var value: DefaultParams2[]

##===----------------------------------------------------------------------===##
# Function positional-only parameters
##===----------------------------------------------------------------------===##

# expected-note @below {{declared here}}
fn has_pos_only[a: Int, b: Int, /, c: Int = 9](): pass

fn test_pos_only():
    # expected-error @below {{positional-only parameter passed as keyword operand: 'b'}}
    has_pos_only[0, b=1, c=2]()
    # expected-error @below {{positional-only parameters passed as keyword operands: 'a', 'b'}}
    has_pos_only[b=1, a=3, c=2]()

    # expected-error @below {{invalid call to 'has_pos_only': could not deduce parameter 'b' of callee 'has_pos_only'}}
    # expected-note @below {{failed to infer parameter 'b', parameter isn't used in any argument}}
    has_pos_only[1, c=9]()

fn indirect_callable_pos_only[
    callable: fn[a: Int, b: Int, /, c: Int = 9] () -> None
]():
    # expected-error @below {{positional-only parameter passed as keyword operand: 'b'}}
    _ = callable[0, b=1, c=2]
    # expected-error @below {{positional-only parameters passed as keyword operands: 'a', 'b'}}
    _ = callable[b=1, a=3, c=2]

##===----------------------------------------------------------------------===##
# Struct keyword parameters
##===----------------------------------------------------------------------===##

# expected-note @+2 {{declared here}}
@value
struct KwParamStruct[a: Int, b: Int = 0]: pass

# expected-note @+2 {{declared here}}
@value
struct VarParamStruct[s: StringLiteral, *args: Int]: pass

fn test_struct_kw_params():
    _ = KwParamStruct[
      a=42,  # expected-note {{previously specified here}}
      a=43,  # expected-error {{duplicate keyword parameter 'a'}}
    ]()

fn test_struct_kw_params2():
    # expected-error @below {{positional parameter follows keyword parameter}}
    _ = KwParamStruct[a=42, 1]()

fn test_struct_kw_params3():
    # expected-error @below {{unknown keyword parameter: 'args'}}
    _ = VarParamStruct["woof", args=7]
    # expected-error @below {{unknown keyword parameter: 'c'}}
    _ = KwParamStruct[7, c=9]()
    # expected-error @below {{unknown keyword parameters: 'z', 'c'}}
    _ = KwParamStruct[7, z=13, c=9]()
    # expected-error @below {{parameter passed both as positional and keyword operand: 'a'}}
    _ = KwParamStruct[7, b=7, a=9]()


##===----------------------------------------------------------------------===##
# Struct positional-only parameters
##===----------------------------------------------------------------------===##

# expected-note @+2 {{declared here}}
@value
struct PosOnlyStruct[a: Int, b: Int, /, c: Int = 9]:
    pass


fn test_pos_only_struct():
    # expected-error @below {{positional-only parameter passed as keyword operand: 'b'}}
    _ = PosOnlyStruct[0, b=1, c=2]
    # expected-error @below {{positional-only parameters passed as keyword operands: 'a', 'b'}}
    _ = PosOnlyStruct[b=1, a=3, c=2]
    # expected-error @below {{could not deduce parameter 'b' of parent struct 'PosOnlyStruct'}}
    # expected-note @below {{parameter isn't used in any argument}}
    _ = PosOnlyStruct[1, c=9]()


##===----------------------------------------------------------------------===##
# CTAD related errors
##===----------------------------------------------------------------------===##

# expected-note @+1 {{struct declared here}}
struct CtadStruct[a: Int]:
    # expected-note @+2 {{declared here}}
    @staticmethod
    fn foo(): pass

fn test_implicitly_parametric_static_methods_fails():
    # expected-error @below {{could not deduce parameter 'a' of parent struct 'CtadStruct'}}
    # expected-note @below {{parameter isn't used in any argument}}
    CtadStruct.foo[5]()

##===----------------------------------------------------------------------===##
# Parameter inference
##===----------------------------------------------------------------------===##

trait SomeTrait:
    fn requirement(self):
        pass

struct NoTraitsType:
    pass

# expected-note @below {{function declared here}}
fn take_some_trait[T: SomeTrait, //](x: T):
    pass

fn pass_no_traits(x: NoTraitsType):
    # expected-error @below {{invalid call to 'take_some_trait'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'NoTraitsType' does not conform to trait 'SomeTrait'}}
    take_some_trait(x)

@value
@register_passable
struct ParamType[p: Int]:
    pass

struct MemParamType[p: Int]:
    pass

# expected-note @below {{function declared here}}
fn autoparams[a: Int](x: ParamType):
    pass

# expected-note @below {{function declared here}}
fn autoparams_mem(x: MemParamType):
    pass

# expected-note @below {{function declared here}}
fn autoparams_variadic(*x: MemParamType):
    pass

struct InferredParam[p: Int, //, T: AnyTrivialRegType, use: ParamType[p]]:
    pass

struct BindStructField:
    # expected-error @below {{failed to infer parameter 'p'}}
    # expected-note @below {{parameter isn't used in any argument}}
    var value: InferredParam[Int]

fn invalid_params[f: fn(ParamType) -> None]():
  # expected-error @below {{invalid call to 'autoparams': could not deduce parameter 'a' of callee 'autoparams'}}
  # expected-note @below {{failed to infer parameter 'a', parameter isn't used in any argument}}
  autoparams[](ParamType[1]())
  # expected-error @below {{callee expects 1 parameter, but 2 were specified}}
  autoparams[1, 2](ParamType[2]())
  # expected-error @below {{failed to infer implicit parameter 'p' of argument 'x' type 'ParamType'}}
  # expected-note @below {{parameter isn't used in any argument}}
  autoparams[1](1)
  # expected-error @below {{failed to infer implicit parameter}}
  # expected-note @below {{parameter isn't used in any argument}}
  autoparams_mem(1)
  # expected-error @below {{failed to infer implicit parameter}}
  # expected-note @below {{parameter isn't used in any argument}}
  autoparams_variadic(1)

  # expected-error @below {{failed to infer implicit parameter 'p' of argument #0}}
  # expected-note @below {{parameter isn't used in any argument}}
  f(1)
