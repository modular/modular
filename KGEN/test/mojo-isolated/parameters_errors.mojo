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
    a: ParametricOnInt[b],  # expected-error {{use of unknown declaration 'b'}}
    b: ParametricOnInt[a],
]():
    pass


# expected-note @+1 {{'Thing' declared here}}
struct Thing[a: Int, b: Int]:
    pass


fn GoodUseOfThing(a: Thing[4, 5]):
    pass


# expected-error @below {{Thing' expects 0 parameters, but 1 was specified}}
fn MultipleThingMetaparams(a: Thing[1, 2][1]):
    pass


# expected-error @+1 {{cannot implicitly convert 'FloatLiteral[1.5]' value to 'Int'}}
fn WeirdMetaParams(a: Thing[1, 1.5]):
    pass


struct Parameterized[p1: Int]:
    # expected-error @below {{invalid redefinition of 'p2'}}
    # expected-note @below {{previous definition here}}
    fn b[p2: Int, p2: Int, p3: Int](self):  # Cannot shadow parameter names.
        pass

    fn __init__(out self):
        pass

    # expected-note @+2 {{'method' declared here}}
    # expected-note @+1 {{function declared here}}
    fn method[B: Int](self, other: Parameterized[p1 + B]):
        pass


# Test that we support partially bound parameters and diagnose incorrect uses
# of parameters.
fn testTestParamStruct(a: Parameterized[4]):
    a.method[7](Parameterized[11]())
    # expected-error-re @below {{invalid call to 'method': method argument #0 cannot be converted from 'Parameterized[{{.*}}12{{.*}}]' to 'Parameterized[{{.*}}11{{.*}}]'}}
    a.method[7](Parameterized[12]())
    a.method[2](Parameterized[6]())
    # expected-error @below {{'method' expects 2 parameters, but 3 were specified}}
    a.method[2, 7]

    # expected-error @below {{'Thing' missing required parameter 'b'}}
    var partial_var_type: Thing[1]


alias DType = __mlir_type.`!kgen.dtype`


# expected-note @below {{struct declared here}}
struct MySIMD[size: Int, type: DType]:
    # expected-note @below {{function declared here}}
    fn __add__(self, rhs: MySIMD[size, type]):
        pass


# expected-note @below {{function declared here}}
fn twoUses[
    dt1: DType, dt2: DType, size: Int
](lhs: MySIMD[size, dt1], rhs: MySIMD[size, dt2]):
    pass


fn testSIMD(
    a: MySIMD[1, __mlir_attr.`#kgen.dtype.constant<f64> : !kgen.dtype`],
    b: MySIMD[2, __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`],
):
    var x = a + a
    var y = b + b
    # expected-error @below {{invalid call to '__add__': could not deduce parameter 'size' of parent struct 'MySIMD'}}
    # expected-note @below {{failed to infer parameter #0, parameter inferred to two different values: '2' and '1'}}
    var z = b + a

    # expected-error @below {{invalid call to 'twoUses': could not deduce parameter 'size' of callee 'twoUses'}}
    # expected-note @below {{failed to infer parameter 'size', parameter inferred to two different values: '1' and '2'}}
    twoUses(a, b)


struct TwoParams[a: Index, b: Index]:
    @implicit
    fn __init__(out self, other: TwoParams[`1`, `1`]):
        pass


# expected-note @below {{function declared here}}
fn infer_then_convert[
    a: Index, b: Index
](lhs: TwoParams[a, b], rhs: TwoParams[a, b]):
    pass


fn left_to_right_implicit_conversion(
    lhs: TwoParams[`1`, `2`], rhs: TwoParams[`1`, `1`]
):
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
def generic_fn[a: FloatLiteral, b: Int](c: Int):
    pass


def call_generic[dt: FloatLiteral]():
    # expected-error @+1 {{invalid call to 'generic_fn': callee expects 2 parameters, but 3 were specified}}
    generic_fn[dt, 1, 42](57)


fn meta_param_then_param_redef[
    dt: __mlir_type.index  # expected-note {{previous definition here}}
](dt: __mlir_type.index):  # expected-error {{invalid redefinition of 'dt'}}
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


fn variadic_int_params[*a: Int]():
    pass


fn callVariadic():
    # expected-error @below {{cannot implicitly convert 'FloatLiteral[1]' value to 'Int'}}
    variadic_int_params[1.0]()


# expected-note @below {{'StructWithVariadic' declared here}}
struct StructWithVariadic[*a: Index]:
    pass


# expected-error @below {{unbound syntax (i.e. `_`) cannot be passed as a variadic parameter}}
fn unbind_variadic(x: StructWithVariadic[_]):
    pass


##===----------------------------------------------------------------------===##
# Function Overloading on Parameters
##===----------------------------------------------------------------------===##


struct ConvertibleFromInt:
    @implicit
    fn __init__(out self, value: Int):
        pass


struct AlsoConvertibleFromInt:
    @implicit
    fn __init__(out self, value: Int):
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
fn arg_overloading_with_param[param: Int]():
    pass


# expected-note @below {{candidate declared here}}
fn arg_overloading_with_param[param: Int](a: Int):
    pass


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
    # expected-error @+1 {{only traits may contain an alias without an initializer}}
    alias MissingTypeAndInit

    # expected-error @below {{only traits may contain an alias without an initializer}}
    alias MissingInit: Int

    # expected-error @+1 {{cannot use a dynamic value in alias initializer}}
    alias NotConstant = variable + 2


fn testConversionQoI():
    # expected-error @+1 {{cannot implicitly convert 'FloatLiteral[1.2]' value to 'Int'}}
    alias intVal: Int = 1.2


@always_inline("nodebug")
fn crash1_callee(
    a: __mlir_type.index, rhs: __mlir_type.index
) -> __mlir_type.index:
    return __mlir_op.`index.add`(a, rhs)


fn crash1_caller[p: __mlir_type.index](a: __mlir_type.index):
    # expected-error @below {{cannot use a dynamic value in alias initializer}}
    alias y = crash1_callee(a, p)


@fieldwise_init
struct StructWithParams[a: Int, b: Int]:
    alias a1 = StructWithParams[1, 2]()
    alias a2 = a+1
    alias a3 = a+b+1

fn testStructWithParams():
    # These are ok because the referenced alias doesn't depend on unbound parameters.
    _ = StructWithParams.a1
    _ = StructWithParams[1].a2
    _ = StructWithParams[1, 2].a3

    # This is an error because the referenced alias depends on an unbound parameter.
    # expected-error @+1 {{cannot access alias 'a3' with unbound parameter 'StructWithParams.b'}}
    _ = StructWithParams[1].a3

    # expected-error @+1 {{cannot access alias 'a3' with unbound parameter 'StructWithParams.a'}}
    _ = StructWithParams.a3




##===----------------------------------------------------------------------===##
# Default struct parameters
##===----------------------------------------------------------------------===##


# expected-error @below {{required positional parameter follows optional positional parameter}}
struct DefaultParams[a: Int, b: Int = 7, msg: Int]:
    pass


@fieldwise_init
struct DefaultParams2[a: Int, b: Int = 7]:  # expected-note {{declared here}}
    pass


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
fn has_pos_only[a: Int, b: Int, /, c: Int = 9]():
    pass


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
@fieldwise_init
struct KwParamStruct[a: Int, b: Int = 0]:
    pass


# expected-note @+2 {{declared here}}
@fieldwise_init
struct VarParamStruct[s: StringLiteral, *args: Int]:
    pass


fn test_struct_kw_params():
    _ = KwParamStruct[
        a=42,  # expected-note {{previously specified here}}
        a=43,  # expected-error {{duplicate keyword parameter 'a'}}
    ]()


fn test_struct_kw_params2():
    # expected-error @below {{positional parameter follows keyword parameter}}
    _ = KwParamStruct[b=42, 1]()


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
@fieldwise_init
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
    fn foo():
        pass


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


@fieldwise_init
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


# expected-note @below {{declared here}}
struct MultiInferred[p: Int, q: Int, //, uP: ParamType[p], uQ: ParamType[q]]:
    pass


struct BindStructField:
    # expected-error @below {{failed to infer parameter 'p'}}
    # expected-note @below {{parameter isn't used in any argument}}
    var value: InferredParam[Int]
    # expected-error @below {{'InferredParam' missing required parameter 'T'}}
    var infer_keyword: InferredParam[p=1]
    # expected-error @below {{inferred parameter passed out of order: 'p'}}
    var multi_infer_ooo: MultiInferred[q=1, p=2]
    # expected-error @below {{inferred parameter passed out of order: 'q'}}
    var multi_infer_ooo2: MultiInferred[p=1, uP=ParamType[1](), q=2]


fn invalid_params[f: fn (ParamType) -> None]():
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


# expected-note @below {{declared here}}
fn substitution_edge_case[p: Int, //, f: fn[a: Int] () [_] -> ParamType[a]]():
    # expected-error @below {{'substitution_edge_case' parameter #2 has 'fn[Int]() -> ParamType[$0]' type, but value has type 'index'}}
    substitution_edge_case[`0`]



# MOCO-846: bad message when types don't match due to parameter expressions
# that can't be evaluated at overload resolution time.
struct HasSize[size: Int]:
    fn __init__(out self):
        pass

# expected-note @below {{function declared here}}
fn has_expr_for_elaborator[width: Int](x: HasSize[width + 4]):
    pass

fn use_take_args[width: Int]():
    # expected-error @below {{argument #0 cannot be converted from 'HasSize[(width + 5)]' to 'HasSize[(width + 4)]'}}
    _ = has_expr_for_elaborator[width](HasSize[size=width + 5]())


# MOCO-1480: handle init-self param not deduce-able.
struct UnusedInitSelfParam[A: Int]:
    # expected-note @below {{function declared here}}
    fn __init__[B: Int](out self: UnusedInitSelfParam[B]):
        pass

fn unused_init_self_param():
    # expected-error @below {{invalid initialization: could not deduce parameter 'B' of callee '__init__'}}
    # expected-note @below {{failed to infer parameter 'B', parameter isn't used in any argument}}
    var slice = UnusedInitSelfParam()


@register_passable("trivial")
struct SimpleSIMD[arg1: Int, size: Int]:
    # expected-note @below {{function declared here}}
    fn __init__[T: AnyType](out self: SimpleSIMD[arg1, 1], value: T): pass

fn dont_miss_inference_conflict(b: SimpleSIMD[40, 1]):
    # expected-error @below {{could not deduce parameter 'T' of callee '__init__'}}
    x = SimpleSIMD[50, 4](b)
