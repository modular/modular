# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | kgen-opt --kgen-print-inline-type-values | FileCheck %s

struct Empty: pass

##===----------------------------------------------------------------------===##
# Input parameters
##===----------------------------------------------------------------------===##

@value
@register_passable("trivial")
struct DType:
    alias type = __mlir_type.`!kgen.dtype`
    var value: Self.type

    alias float32 = __mlir_attr.`#kgen.dtype.constant<f32> : !kgen.dtype`
    alias int32 = __mlir_attr.`#kgen.dtype.constant<si32> : !kgen.dtype`

    @always_inline("builtin")
    @implicit
    fn __init__(out self, value: Self.type):
        self.value = value

# CHECK-LABEL: lit.struct.decl @SIMD
# CHECK-SAMEL <[[SIMDDT:.*]]: !DType, [[SIMDSIZE:.*]]: !Int>
# CHECK-SAME: register_passable
@register_passable("trivial")
struct SIMD[dt: DType, size: Int]:
    var value: __mlir_type[`!pop.simd<`, size.value, `, `, dt.value, `>`]

    fn __add__(lhs, rhs: Self) -> Self:
        while __mlir_attr.true:
            pass

    @staticmethod
    fn splat():
        pass


@register_passable
struct StructWithIntParam[size: Int]:
    pass

# CHECK-LABEL: lit.fn @"paramArith{{.*}}"<x: !Int>() -> !kgen.none
fn paramArith[x: Int]():
    # CHECK: lit.alias.decl *"y`": !Bool = <{value: i1 = eq(#lit.struct.extract<:!Int x, "value">, 99)}>
    alias y = x == 98 + 1

fn take_3index(a: Int, b: Int, c: Int) -> Int:
    return a

# CHECK-LABEL: lit.fn @"fancy_signature{{.*}}"<dt: !DType, size: !Int>
# CHECK-SAME: (%x: {{.*}}#SIMD <:!DType dt, :!Int size>{{.*}}>,
# CHECK-SAME: %exp: {{.*}}#SIMD <:!DType dt, :!Int size>{{.*}}>) -> !Int
fn fancy_signature[dt: DType, size: Int](
    x: SIMD[dt, size],
    exp: (SIMD)[dt, size]
) -> Int:
  # CHECK: %local = lit.var.decl "local" var
  # CHECK: %[[TMP1:.*]] = kgen.param.constant: !Int = <size>
  # CHECK: %[[TMP2:.*]] = kgen.param.constant: !Int = <size>
  # CHECK: %[[TMP3:.*]] = kgen.param.constant: !Int = <size>
  # CHECK: %[[RES:.*]] = lit.call @parameters::@"take_3index{{.*}}(%[[TMP1]], %[[TMP2]], %[[TMP3]])
  # CHECK: lit.ref.store %[[RES]], %local
  var local = take_3index(size, size, size)

  # CHECK: %[[TMP:.*]] = kgen.param.constant: !Int = <{value = add(#lit.struct.extract<:!Int size, "value">, 42)}>
  # CHECK: lit.return %[[TMP]]
  return size+42


fn generic_fn[a: DType, b: Int, c: __mlir_type.`!kgen.type`](d : Int):
  pass

# CHECK: lit.fn @"call_generic{{.*}}"<dt: !DType>()
fn call_generic[dt: DType]():
  # CHECK: %[[C57:.*]] = {{.*}}constant{{.*}}57
  # CHECK: lit.call @parameters::@"generic_fn{{.*}}"<:!DType dt, :!Int {{.*}}42{{.*}}, :type !DType>(%[[C57]])
  generic_fn[dt, 42, DType](57)

  # CHECK: %[[C57_2:.*]] = {{.*}}constant{{.*}}57
  # CHECK: lit.call @parameters::@"generic_fn{{.*}}"<:!DType dt, :!Int {13}, :type @parameters::@SIMD<{{.*}}:!DType dt, :!Int {4}>{{.*}}>(%[[C57_2]])
  generic_fn[dt, 13, SIMD[dt, 4]](57)

# CHECK-LABEL: lit.struct.decl @TestParamStruct<
# CHECK-SAME: [[A:.*]]: !Int>
@value
@register_passable("trivial")
struct TestParamStruct[A: Int]:

  # CHECK: lit.fn @"method{{.*}}"<B: !Int>(%self: !lit.struct<#TestParamStruct <:!Int [[A]]>
  # CHECK-SAME: %other: {{.*}}#TestParamStruct <:!Int {value = add(#lit.struct.extract<:!Int [[A]], "value">, #lit.struct.extract<:!Int B, "value">)}>>
  fn method[B: Int](self: TestParamStruct[A], other: TestParamStruct[A+B]):
    pass

  # CHECK-LABEL: lit.fn @"aliases{{.*}}%x: {{.*}}#TestParamStruct <
  fn aliases(self, x: TestParamStruct[TestParamStruct[A].TypeLevelAlias]):
    # CHECK: lit.alias.decl [[B:.*]]: !Int = <{value = add(mul(#lit.struct.extract<:!Int *"A`", "value">, 2), 1)}>
    alias B = A+A+1
    # CHECK: lit.alias.decl *"C{{.*}}: !Int = <{value = add(mul(#lit.struct.extract<:!Int *"A`", "value">, 3), 1)}>
    alias C = B+A
    # CHECK: lit.alias.decl [[D:.*]]: {{.*}}@TestParamStruct<:!Int {{.*}}1{{.*}}> =
    # CHECK-SAME: <apply(:!lit.generator<{{.*}}TestParamStruct <:!Int {1}>>> {{.*}}__init__()"<:!Int {1}>)>
    alias D = TestParamStruct[1]()
    # CHECK: %temp = lit.var.decl {{.*}} : {{.*}}@TestParamStruct<:!Int
    var temp: TestParamStruct[C]

    # CHECK: lit.alias.decl *"intVal{{.*}}": !Int = <{42}>
    alias intVal : Int = 42

    # CHECK: %temp2 = lit.var.decl {{.*}} : {{.*}}@TestParamStruct<:!Int {value = mul(#lit.struct.extract<:!Int [[A]], "value">, 2)}>
    var temp2: TestParamStruct[TestParamStruct[A].TypeLevelAlias]

  # CHECK: lit.alias.decl *"TypeLevelAlias{{.*}}": !Int = <{value = mul(#lit.struct.extract<:!Int *"A`", "value">, 2)}>
  alias TypeLevelAlias = A+A

# Test that we support partially bound parameters.
# CHECK-LABEL: lit.fn @"testTestParamStruct
fn testTestParamStruct(a: TestParamStruct[4]):
  # CHECK: %arg11 = lit.var.decl {{.*}} : {{.*}}@TestParamStruct<:!Int {{.*}}11
  # CHECK: %0 = lit.call {{.*}}@TestParamStruct::@"__init__{{.*}}<:!Int {{.*}}11{{.*}}>()
  var arg11 = TestParamStruct[11]()

  # CHECK: %1 = lit.ref.load %arg11
  # CHECK: lit.call {{.*}}@TestParamStruct::@"method{{.*}}<:!Int {4}, :!Int {7}>(%a, %1)
  a.method[7](arg11)

# CHECK-LABEL: lit.fn @"testSIMD(
fn testSIMD(a: SIMD[DType.float32, 1],
            b: SIMD[DType.int32, 1],
            mut reff: SIMD[DType.int32, 1]):
  # CHECK: %field1 = lit.var.decl {{.*}} : !lit.ref<scalar<f32>,
  var field1 = a.value
  # CHECK: %field2 = lit.var.decl {{.*}} : !lit.ref<scalar<si32>,
  var field2 = reff.value

  # Test calls to methods and operators on parameterized type.
  # CHECK: lit.call {{.*}}@SIMD::@"__add__{{.*}}<:!DType {:dtype f32}, :!Int {1}>(%a, %a)
  var x = a+a
  # CHECK: lit.call {{.*}}@SIMD::@"__add__{{.*}}<:!DType {:dtype si32}, :!Int {1}>(%b, %b)
  var y = b+b

# Show that forward references of parameter names can be correctly resolved.
#
# CHECK-LABEL: lit.fn @"paramResolution[
# CHECK-SAME: Int,
# CHECK-SAME: parameters::StructWithIntParam[$0],
# CHECK-SAME: Int,
# CHECK-SAME: parameters::StructWithIntParam[$2]
# CHECK-SAME: ]()"<
# CHECK-SAME: size1: !Int, a: @parameters::@StructWithIntParam<:!Int size1>,
# CHECK-SAME: size2: !Int, b: @parameters::@StructWithIntParam<:!Int size2>>()
fn paramResolution[size1: Int, a: StructWithIntParam[size1],
                   size2: Int, b: StructWithIntParam[size2]]():
  pass

# Show that we can implicitly convert from 42's literal type to Int.
# CHECK-LABEL: lit.fn @"implConversion
# CHECK: <a: @parameters::@StructWithIntParam<:!Int {42}>>
fn implConversion[a: StructWithIntParam[42]]():
  pass

# CHECK-LABEL: lit.struct.decl @Pair<dt: !DType>
@register_passable
struct Pair[dt: DType]:
 # CHECK: lit.struct.field a : {{.*}}#SIMD <:!DType dt, :!Int {{.*}}42{{.*}}>{{.*}}>
 # CHECK: lit.struct.field b : !Int
  var a : SIMD[dt, 42]
  var b : Int

  # CHECK: lit.fn @"__init__{{.*}}-> !lit.struct<#Pair <:!DType dt>>
  @implicit
  fn __init__(out self, a: SIMD[dt, 42]):
    self.a = a
    self.b = 4
  # CHECK: }

  fn __copyinit__(out self, existing: Self): pass

# CHECK: }

# CHECK: useParameterizedField
fn useParameterizedField[x: Pair[DType.float32]]():
  # CHECK: lit.alias.decl *"y{{.*}}":
  alias y : SIMD[DType.float32, 42] = x.a


# CHECK-LABEL: lit.struct.decl @TypeParameter
# CHECK-SAME: <[[TYPE:.*]]: type>
struct TypeParameter[T: __mlir_type.`!kgen.type`]:
  # CHECK: @"bar(parameters::TypeParameter{{.*}}(%self: {{.*}} read_mem, %val: !kgen.param<[[TYPE]]>)
  fn bar(self, val: T):
    pass

# Test that parameter decls can refine subsequent ones in the same param list.
# CHECK-LABEL: lit.struct.decl @ParamSubst
# CHECK-SAME: <[[TYPE:.*]]: type, shape: variadic<[[TYPE]]>>
struct ParamSubst[
    T: AnyTrivialRegType,
    shape: __mlir_type[`!kgen.variadic<`, T,`>`],
  ]: pass

# CHECK-LABEL: lit.fn @"testParamSubst
fn testParamSubst():
  # CHECK: %xx = lit.var.decl {{.*}} : !lit.ref<@parameters::@ParamSubst<:type index, :variadic<index> [1, 2]>
  var xx : ParamSubst[Index, __mlir_attr.`#kgen.variadic<1, 2> : !kgen.variadic<index>`]


# Test parameter substitution.
# CHECK-LABEL: lit.fn @"fnToCall{{.*}}"<size, arr: array<size, f32>>()
fn fnToCall[size: Index, arr: __mlir_type[`!pop.array<`, size, `, f32>`]]():
  pass

# CHECK: lit.fn @"fnWithCall{{.*}}"<array: array<10, f32>
fn fnWithCall[array: __mlir_type[`!pop.array<10, f32>`]]():
   # CHECK: lit.call @parameters::@"fnToCall{{.*}}"<10, :array<10, f32> array>()
   fnToCall[Int(10).value, array]()

# CHECK-LABEL: lit.fn @"meta_str{{.*}}"<[""]*"value`": string, +, type: @stdlib::@builtin::@stubs::@StringLiteral<:string *"value`">>() -> !kgen.none
fn meta_str[type: StringLiteral]():
  pass

# CHECK-LABEL: lit.fn @"str_input_param()"() -> !kgen.none
fn str_input_param():
  # CHECK: %0 = lit.call @parameters::@"meta_str{{.*}}"<{{.*}}@StringLiteral<:string "123">{{.*}}>()
  meta_str["123"]()

@value
@register_passable("trivial")
struct TwoParams[a: Int, b: Int]:
    pass

# CHECK-LABEL: lit.fn @"signature_capture{{.*}}"<
# CHECK-SAME: a: !Int,
# CHECK-SAME: f: !lit.generator<<"b": !Int>() -> {{.*}}TwoParams <:!Int a, :!Int *(0,0)>{{.*}}>
fn signature_capture[a: Int, f: fn[b: Int]() -> TwoParams[a, b]]():
    _ = f[2]()

# CHECK-LABEL: lit.fn @"my_constrained{{.*}}"<{{.*}}cond: !Bool, message: {{.*}}@StringLiteral<:string *"value`">>()
fn my_constrained[cond: Bool, message: StringLiteral]():
    # CHECK: kgen.param.assert <#lit.struct.extract<:!Bool cond, "value">>, *"value`"
    __mlir_op.`kgen.param.assert`[cond=cond.__mlir_i1__(), message=message.value]()
    return


# CHECK-LABEL: lit.fn @"pass_str_param
fn pass_str_param():
    # CHECK: lit.call {{.+}}my_constrained{{.*}}"<{{.*}}:!Bool {:i1 1}, {{.*}}StringLiteral<:string "foo">{{.*}}>()
    my_constrained[1==1, "foo"]()

# CHECK-LABEL: lit.fn @"implicit_params
# CHECK-SAME: <?, [[VALUE0:.*]]: !Int, [[VALUE1:.*]]: !Int>
# CHECK-SAME: %value: {{.*}}#TwoParams <:!Int [[VALUE0]], :!Int [[VALUE1]]>
fn implicit_params(value: TwoParams):
    pass

# CHECK-LABEL: lit.fn @"implicit_params_with_others
# CHECK-SAME: <a: !Int, ?, [[LHS0:.*]]: !Int, [[LHS1:.*]]: !Int, [[RHS0:.*]]: !Int, [[RHS1:.*]]: !Int>
# CHECK-SAME: %lhs: {{.*}}#TwoParams <:!Int [[LHS0]], :!Int [[LHS1]]>
# CHECK-SAME: %rhs: {{.*}}#TwoParams <:!Int [[RHS0]], :!Int [[RHS1]]>
fn implicit_params_with_others[a: Int](lhs: TwoParams, rhs: TwoParams):
    pass

# CHECK-LABEL: lit.fn @"infer_implicit_params()"
fn infer_implicit_params():
    # CHECK: call {{.*}}implicit_params{{.*}}<:!Int {1}, :!Int {2}
    var one = TwoParams[1, 2]()
    implicit_params(one)
    var two = TwoParams[3, 4]()
    # CHECK: call {{.*}}implicit_params_with_others{{.*}}<:!Int {42},
    # CHECK-SAME: :!Int {1}, :!Int {2}, :!Int {3}, :!Int {4}>
    implicit_params_with_others[42](one, two)

    # CHECK: alias.decl *"partial_bind{{.*}}: !lit.generator<<?, !Int, !Int, !Int, !Int>
    # CHECK-SAME: implicit_params_with_others{{.*}}<:!Int {1}, :!Int ?, :!Int ?, :!Int ?, :!Int ?>
    alias partial_bind = implicit_params_with_others[1]
    # CHECK: lit.call {{.*}}implicit_params_with_others{{.*}}<:!Int {1}, :!Int {1}, :!Int {2}, :!Int {3}, :!Int {4}>
    partial_bind(one, two)

fn implicit_params_with_var_params[*Ts: Int](s: TwoParams[1]): pass

# CHECK-LABEL: lit.fn @"test_implicit_params_with_var_params
fn test_implicit_params_with_var_params():
    # CHECK: [[VAL0:%.*]] = lit.call {{.*}}@TwoParams::@"__init__{{.*}}<:!Int {1}, :!Int {2}>() :
    # CHECK: call {{.*}}@"implicit_params_with_var_params{{.*}}<:variadic<!Int> [], :!Int {2}>([[VAL0]])
    implicit_params_with_var_params(TwoParams[1, 2]())

# CHECK-LABEL: lit.fn @"explicit_autoparameterization
# CHECK-SAME: "<?, [[V0:.*]]: !Int, [[W0:.*]]: !Int, [[W1:.*]]: !Int>(
# CHECK-SAME: %v: {{.*}}#TwoParams <:!Int {5}, :!Int [[V0]]>
# CHECK-SAME: %w: {{.*}}#TwoParams <:!Int [[W0]], :!Int [[W1]]>
fn explicit_autoparameterization(v: TwoParams[5, _], w: TwoParams[b=_, a=_]):
    pass

@register_passable("trivial")
struct IndexParam[x: Index]:
    @implicit
    fn __init__(out self, p: __mlir_type.`!kgen.none`):
        pass


# CHECK-LABEL: lit.fn @"autoparam_of_params
# CHECK-SAME: <[""]*"x`", a, +, b: {{.*}}IndexParam<*"x`">, c: {{.*}}IndexParam<a>
fn autoparam_of_params[a: Index, //, b: IndexParam, c: IndexParam[a]]():
    pass

# CHECK-LABEL: lit.fn @"autoparam_of_struct_metatype_params
# CHECK-SAME: <[""]*"x`1", +, a: meta<!lit.struct<#IndexParam <*"x`1">>>>
fn autoparam_of_struct_metatype_params[a: __type_of(IndexParam)]():
    pass

@value
@register_passable("trivial")
struct DependentParams[x: Index, //, p: IndexParam[x]]:
    pass


# CHECK-LABEL: lit.fn @"autoparam_of_dependent_params
# CHECK-SAME: <[""]*"x`", [""]*"p`1": {{.*}}IndexParam<*"x`">, +, dp: {{.*}}DependentParams<*"x`", :{{.*}}IndexParam<*"x`"> *"p`1">>
fn autoparam_of_dependent_params[dp: DependentParams]():
    pass


# CHECK-LABEL: lit.fn @"function_autoparam
# CHECK-SAME: :{mut |*(0,0)|, mut |*(0,1)|}:<[""][[G_LT:.*]]: origin.set, [""][[F_LT:.*]]: origin.set, +
# CHECK-SAME: f: !lit.generator<:[[F_LT]]:() capturing -> !kgen.none>
# CHECK-SAME: g: !lit.generator<:[[G_LT]]:() capturing -> !kgen.none>
fn function_autoparam[f: fn () capturing [_] -> None, g: fn () capturing [_] -> None]():
    @parameter
    fn function():
        pass

    # CHECK: lit.alias.decl *"bind_one{{.*}}": !lit.generator<() capturing -> !kgen.none> =
    # CHECK-SAME: <{{.*}}function_autoparam{{.*}}<:origin.set {}, :origin.set {}, :{{.*}} *"function()", :{{.*}} *"function()">
    alias bind_one = function_autoparam[function, function]


# CHECK-LABEL: lit.fn @"nonprop_capture_set
# CHECK-SAME: ()"<f: !lit.generator<<origin.set, +, "g": !lit.generator<:*(1,0):() capturing -> !kgen.none>>:*(0,0):() -> !kgen.none>>()
fn nonprop_capture_set[f: fn[g: fn () capturing [_] -> None] () -> None]():
    pass


# CHECK-LABEL: lit.fn @"autoparam_param_vararg
# CHECK-SAME: <[""]*"__origins__`": origin.set, +, f: {{.*}}, x: variadic<index> var>
fn autoparam_param_vararg[f: fn () [_] -> None, *x: Index]():
    pass


# CHECK-LABEL: lit.fn @"auto_kw_default{{.*}}"<u = 3, |, v = 3, ?, {{.*}}, {{.*}}>(%a
fn auto_kw_default[u: Index = `3`, /, v: Index = `3`](a: IndexParam, b: IndexParam):
  pass


# CHECK-LABEL: lit.fn @"test_auto_kw_default
# CHECK-SAME: <?, [[A:.*]], [[B:.*]]>(%a
fn test_auto_kw_default(a: IndexParam, b: IndexParam):
  # CHECK-NEXT: <3, 3, [[A]], [[B]]>
  auto_kw_default(a, b)
  # CHECK-NEXT: <1, 3, [[A]], [[B]]>
  auto_kw_default[`1`](a, b)
  # CHECK-NEXT: <3, 2, [[A]], [[B]]>
  auto_kw_default[v=`2`](a, b)
  # CHECK-NEXT: <1, 2, [[A]], [[B]]>
  auto_kw_default[`1`, v=`2`](a, b)


trait ASuperTrait:
    pass


trait ASubTrait(ASuperTrait):
    pass


struct StructWithTraitParam[T: ASuperTrait]():
    pass

    fn __init__(out self: StructWithTraitParam[T]):
        pass


# CHECK-LABEL: lit.fn @"test_upcast_trait
fn test_upcast_trait[T: ASubTrait](tuples: StructWithTraitParam[T]):
    pass


##===----------------------------------------------------------------------===##
# Memory-only parameters
##===----------------------------------------------------------------------===##

@value
struct MemoryType:
    var value: Int

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        self.value = value

struct NonMovableMemoryType:
    var value: Int

    @always_inline
    @implicit
    fn __init__(out self, value: Int):
        self.value = value

fn makeMemoryValue(x: Int) -> MemoryType:
    return x

fn passMemoryValue(x: MemoryType) -> MemoryType:
    return x

@always_inline
fn readMemoryValue(x: NonMovableMemoryType) -> Int:
    return x.value

# CHECK-LABEL: lit.fn @"callMemoryValueParam
fn callMemoryValueParam():
    # CHECK: lit.alias.decl [[PARAM_VALUE1:.*]]: {{.*}}MemoryType = <apply_result_slot({{.*}}makeMemoryValue{{.*}}, {{.*}}1234
    alias paramValue = makeMemoryValue(1234)
    # CHECK: %dynamicLet = lit.var.decl
    # CHECK: %[[PARAM_VALUE2:.*]] = kgen.param.materialize: !MemoryType =
    # CHECK: lit.ref.store %[[PARAM_VALUE2]], %dynamicLet
    var dynamicLet = paramValue

    alias nonMovable = NonMovableMemoryType(42)
    # CHECK: %dynamicVar = lit.var.decl
    # CHECK: %[[NON_MOVABLE:.*]] = kgen.param.materialize: !NonMovableMemoryType
    # CHECK: lit.ref.store %[[NON_MOVABLE]], %dynamicVar
    var dynamicVar = nonMovable

    # CHECK: lit.alias.decl [[COPY:.*]]: {{.*}}MemoryType = <apply_result_slot({{.*}}passMemoryValue{{.*}} store_to_mem(
    alias copy = passMemoryValue(paramValue)
    # CHECK: [[MVALUE:%.*]] = lit.var.decl "anonymous*"
    # CHECK: [[PVALUE:%.*]] = kgen.param.materialize: !MemoryType =
    # CHECK: lit.ref.store [[PVALUE]], [[MVALUE]]
    # CHECK-NEXT: [[IMMREF:%.*]] = lit.ref.immut [[MVALUE]]
    # CHECK: lit.var.decl
    # CHECK: lit.call {{.*}}passMemoryValue{{.*}}([[IMMREF]], %{{.*}})
    _ = passMemoryValue(copy)

    # CHECK: lit.call {{.*}}MemoryType::@"__init__(::Int)"), {22})>
    memoryParam[MemoryType(22)]()

    # CHECK: dontFoldMemoryCall{{.*}}{42})))
    alias dontFoldMemoryCall = readMemoryValue(NonMovableMemoryType(42)).value

# CHECK-LABEL: lit.fn @"memoryParam{{.*}}"<value: !MemoryType>()
fn memoryParam[value: MemoryType]():
    pass

@register_passable("trivial")
struct InitSelfCtor:
    var x: Int

    @always_inline("builtin")
    @implicit
    fn __init__(out self, x: Int):
        self.x = x

    @always_inline("builtin")
    fn __add__(self, rhs: Self) -> Self:
        return self.x + rhs.x

@register_passable("trivial")
struct InitSelfParam[x: InitSelfCtor]:
    pass


@value
struct IntBox:
    var x: Int

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, value: Int):
        self.x = value


@always_inline
fn intbox_memory_result(x: Int) -> IntBox:
    return x


# CHECK-LABEL: lit.fn @"interpret_initself_ctor
# CHECK-SAME: %arg: !lit.struct<#InitSelfParam <:!InitSelfCtor {x: !Int = {42}}>>
fn interpret_initself_ctor(arg: InitSelfParam[InitSelfCtor(42)]):
    # CHECK-NEXT: !lit.generator<() -> !lit.struct<#InitSelfParam <:!InitSelfCtor {x: !Int = {3}}>>>
    alias refined_fn = refine_memory_only_results[1, 2]

    # CHECK: [[CST:%.*]] = kgen.param.constant: !InitSelfCtor = <{x: !Int = {42}}>
    # CHECK-NEXT: store [[CST]], %inlined_initself_call
    var inlined_initself_call = InitSelfCtor(42)

    # CHECK: %inlined_byrefresult_call = lit.var.decl "inlined_byrefresult_call"
    # CHECK-NEXT: [[CST:%.*]] = kgen.param.constant: !Int = <{24}>
    # CHECK-NEXT: lit.call{{.*}}intbox_memory_result{{.*}}([[CST]], %inlined_byrefresult_call)
    var inlined_byrefresult_call = intbox_memory_result(24)


fn refine_memory_only_results[a: InitSelfCtor, b: InitSelfCtor]() -> InitSelfParam[a + b]:
    pass


struct ConvertFromIntLiteral:
    @implicit
    fn __init__(out self, x: IntLiteral):
        pass


fn nonmaterializable_arg(x: IntLiteral) -> ConvertFromIntLiteral:
    return x


# CHECK-LABEL: lit.fn @"parameter_memoryonly_call
fn parameter_memoryonly_call():
    # CHECK: [[CST:%.*]] = kgen.param.materialize: !ConvertFromIntLiteral = <apply_result_slot({{.*}}@ConvertFromIntLiteral::@"__init__
    # CHECK-NEXT: store [[CST]], %x
    var x: ConvertFromIntLiteral = 2
    # CHECK: [[CST:%.*]] = kgen.param.materialize: !ConvertFromIntLiteral = <apply_result_slot({{.*}}@"nonmaterializable_arg
    # CHECK-NEXT: store [[CST]], %y
    var y = nonmaterializable_arg(4)


struct IntBoxParam[b: IntBox]: pass
fn takeIntBoxParam[size: IntBox](a: IntBoxParam[size]): pass
fn selectIntBoxFromVariadic(*values: IntBox) -> IntBox: pass


# CHECK-LABEL: lit.fn @"parameter_call_drop_dangling_implicit_origins
fn parameter_call_drop_dangling_implicit_origins[b: IntBox]():
    alias res = selectIntBoxFromVariadic(b)
    var wrapper : IntBoxParam[res]
    takeIntBoxParam[res](wrapper)


##===----------------------------------------------------------------------===##
# First-class functions as parameters.
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"takeCallable{{.*}}"<
# CHECK-SAME: callable: !lit.generator<(index, |) -> index>>(%a: index) -> index
fn takeCallable[
     callable: fn(Index) -> Index
   ](a: Index) -> Index:
  # CHECK-NEXT: %0 = lit.call[!lit.generator<(index, |) -> index>: callable](%a)
  # CHECK-NEXT: lit.return %0
  return callable(a)

fn takeAndReturnIndex(x: Index) -> Index:
  return x

fn posOnlyArg(x: Index, /):
  pass

# CHECK-LABEL: lit.fn @"takeAndReturnIndex
fn passFunction(a: Index) -> Index:
  # CHECK: rebind(:!lit.generator<("x": index, |) -> !kgen.none> {{.*}}posOnlyArg
  alias changeKw: fn(x: Index) -> None = posOnlyArg

  # CHECK: lit.call @parameters::@"takeCallable{{.*}}<:!lit.generator<(index, |) -> index>
  # CHECK-SAME: rebind(:!lit.generator<("x": index) -> index> {{.*}}takeAndReturnIndex{{.*}}")>(%a)
  return takeCallable[takeAndReturnIndex](a)

# CHECK-LABEL: lit.fn @"callableWithParam{{.*}}"<type: dtype>() -> !kgen.none
fn callableWithParam[type: __mlir_type.`!kgen.dtype`]():
  pass

# CHECK-LABEL: lit.fn @"takeCallable2
fn takeCallable2[
      func: fn[dt: __mlir_type.`!kgen.dtype`]() -> None
  ]():
      pass

# CHECK-LABEL: lit.fn @"passFunctionParam2
fn passFunctionParam2():
  # CHECK: lit.call @parameters::@"takeCallable2{{.*}}"<
  # CHECK-SAME: :!lit.generator<<"dt": dtype>() -> !kgen.none> rebind(:!lit.generator<<"type": dtype>() -> !kgen.none> @parameters::@"callableWithParam
  takeCallable2[callableWithParam]()


@register_passable("trivial")
struct ParamType[x: Index]:
    pass


# CHECK-LABEL: lit.fn @"dependent_function_type
fn dependent_function_type[a: Index, f: fn (ParamType[a]) -> None]():
    alias func = dependent_function_type
    # CHECK: lit.call{{.*}}dependent_function_type
    func[a, f]()

fn overloaded_function():
    pass

fn overloaded_function(a: Index):
    pass

struct ParamFuncType[f: fn() -> None]:
    pass

fn bind_twice[f: fn() -> None, g: fn(Index) -> None]():
    pass

fn variadic_func_param[*fs: fn() -> None]():
    pass

# CHECK-LABEL: lit.fn @"bind_overloaded_fn
fn bind_overloaded_fn[f: fn[f: fn () -> None] () -> None]():
    # CHECK-NEXT: meta<!lit.struct<#ParamFuncType <:!lit.generator<() -> !kgen.none> {{.*}}@"overloaded_function()"
    alias T = ParamFuncType[overloaded_function]
    # CHECK-NEXT: meta<!lit.struct<#ParamFuncType <:!lit.generator<() -> !kgen.none> {{.*}}@"overloaded_function()"
    alias U = ParamFuncType[f=overloaded_function]

    # CHECK-NEXT: bind_params(:{{.*}} f, {{.*}}@"overloaded_function()")
    alias g = f[overloaded_function]
    # CHECK-NEXT: bind_params(:{{.*}} f, {{.*}}@"overloaded_function()")
    alias h = f[f=overloaded_function]

    # CHECK-NEXT: bind_twice{{.*}}<:!lit.generator<() -> !kgen.none> {{.*}}@"overloaded_function()", :!lit.generator<(index, |) -> !kgen.none> {{.*}}overloaded_function(__mlir_type.index)")>
    alias bound = bind_twice[overloaded_function][overloaded_function]

    # CHECK-NEXT: variadic_func_param{{.*}}<:variadic<{{.*}}> [{{.*}}@"overloaded_function()", {{.*}}@"overloaded_function()"]>
    alias bind_variadic = variadic_func_param[overloaded_function, overloaded_function]

##===----------------------------------------------------------------------===##
# Alias resolution
##===----------------------------------------------------------------------===##

# CHECK: lit.alias.decl *"boolDtype{{.*}}": dtype = <bool>
alias boolDtype = __mlir_attr.`#kgen.dtype.constant<bool> : !kgen.dtype`
# CHECK: lit.alias.decl *"FOURTY_TWO{{.*}}": {{.*}}<:!pop.int_literal 42>
alias FOURTY_TWO = 42

# CHECK-LABEL: lit.struct.decl @A
# CHECK-SAME: <v: !Int>
struct A[v: Int]:
  # CHECK: lit.alias.decl *"member{{.*}}": !Int = <{value = add(#lit.struct.extract<:!Int v, "value">, 42)}>
  alias member = v + FOURTY_TWO

# CHECK-LABEL: lit.fn @"testUseOfAliases
fn testUseOfAliases():
  # This type checks.
  SIMD[DType(boolDtype), 4].splat()
  # CHECK: lit.alias.decl *"y{{.*}}": !Int = <{{.*}}44
  alias y = A[2].member

@register_passable
struct MyDType:
  var state : Index

  fn __copyinit__(out self, existing: Self):
    self.state = self.state

  @implicit
  fn __init__(out self, value: Index):
     self.state = value

  fn __eq__(self, rhs: MyDType) -> Bool:
     return __mlir_attr.true

  alias ui8 = MyDType(Int(1).value)
  alias float32 = MyDType(Int(2).value)
  alias float64 = MyDType(Int(3).value)

struct MyVector[size: Int, dtype: MyDType]:
    pass

fn testMyDType[dt: MyDType](a: MyVector[4, MyDType.float32],
                            b: MyVector[4, dt]):
    pass

# Issue #6828: Unqualified name lookup into structs doesn't work
# CHECK-LABEL: lit.struct.decl @UnqualAliasLookup<param: !Int>
struct UnqualAliasLookup[param: Int]:
  # CHECK: lit.alias.decl *"member{{.*}}": !Int = <{value = add(#lit.struct.extract<:!Int param, "value">, 1)}>
  alias member = param+1
  fn get(self) -> Int:
    # CHECK: %0 = kgen.param.constant: !Int = <{value = add(#lit.struct.extract<:!Int param, "value">, 1)}>
    return Self.member

##===----------------------------------------------------------------------===##
# Variadic parameters
##===----------------------------------------------------------------------===##

# CHECK-LABEL: lit.fn @"fnWithVariadics{{.*}}"<b: variadic<!Int> var>
fn fnWithVariadics[*b: Int]():
  pass

# CHECK-LABEL: lit.struct.decl @StructWithVariadics<b: variadic<!Int> var>
struct StructWithVariadics[*b: Int]:
    @implicit
    fn __init__(out self, i: Int):
        pass

# CHECK-LABEL: lit.fn @"useParamVariadics
fn useParamVariadics():
  # CHECK-NEXT: lit.call @parameters::@"fnWithVariadics{{.*}}"<:variadic<!Int> []>()
  fnWithVariadics()

  # CHECK: lit.call @parameters::@"fnWithVariadics{{.*}}"<:variadic<!Int> [{1}]>()
  fnWithVariadics[1]()
  # CHECK: lit.call @parameters::@"fnWithVariadics{{.*}}"<:variadic<!Int> [{1}, {2}]>()
  fnWithVariadics[1, 2]()

  # This keeps the parameters unbound, allowing them to be used with different length..
  # CHECK-NEXT: lit.alias.decl *"fnAlias{{.*}}": !lit.generator<<"b": variadic<!Int> var>() -> !kgen.none>
  # CHECK-SAME: = <@parameters::@"fnWithVariadics{{.*}}">
  alias fnAlias = fnWithVariadics

  # Use of an unbound thing in a DRValue context binds an empty variadic list.
  # FIXME(#29495): Pack references aren't working right.
  # HECK-NEXT: [[TMP:%.*]] = kgen.create_closure[!lit.generator<() -> !kgen.none>: @parameters::@"fnWithVariadics{{.*}}"<:variadic<!Int> []>]()
  # HECK-NEXT: %fnLet = lit.var.decl "fnLet" : {{.*}}!lit.generator<() -> !kgen.none>
  # HECK-NEXT: lit.ref.store [[TMP]], %fnLet
  # var fnLet = fnWithVariadics

  # CHECK-NEXT: %a = lit.var.decl {{.*}} : !lit.ref<@{{.*}}::@StructWithVariadics<:variadic<!Int> []>
  var a: StructWithVariadics
  # CHECK-NEXT: %b = lit.var.decl {{.*}} : !lit.ref<@{{.*}}::@StructWithVariadics<:variadic<!Int> [{1}]>
  var b: StructWithVariadics[1]
  # CHECK-NEXT: %c = lit.var.decl {{.*}} : !lit.ref<@{{.*}}::@StructWithVariadics<:variadic<!Int> [{1}, {2}]>
  var c: StructWithVariadics[1, 2]

  # TODO(16040): fix symbol name mangling to erase parameter name 'b'
  # CHECK: lit.call {{.*}}@StructWithVariadics::@"__init__({{.*}}<:variadic<!Int> [{1}]>
  var d = StructWithVariadics[1](2)
  # CHECK: lit.call {{.*}}@StructWithVariadics::@"__init__({{.*}}<:variadic<!Int> []>
  var e = StructWithVariadics(3)


# CHECK-LABEL: lit.fn @"unpack_variadic
fn unpack_variadic[*a: Int]():
    # CHECK-NEXT: @StructWithVariadics<:variadic<!Int> a>
    alias T = StructWithVariadics[*a]
    # CHECK-NEXT: fnWithVariadics{{.*}}<:variadic<!Int> a>
    alias f = fnWithVariadics[*a]


# CHECK-LABEL: lit.fn @"variadic_parameter{{.*}}"<elems: variadic<index>>
fn variadic_parameter[elems: __mlir_type.`!kgen.variadic<index>`]() -> Int:
    return 3

fn dependent_variadic_parameter[
    type: __mlir_type.`!kgen.type`, *values: type
](): pass

# CHECK-LABEL: lit.fn @"pass_variadic{{.*}}"<elems: variadic<index>>
fn pass_variadic[elems: __mlir_type.`!kgen.variadic<index>`]():
    # CHECK-NEXT: lit.call @parameters::@"variadic_parameter{{.*}}"<:variadic<index> elems>
    _ = variadic_parameter[elems]()
    # CHECK: lit.call @parameters::@"dependent_variadic_parameter{{.*}}"<:type !Int, :variadic<!Int>
    _ = dependent_variadic_parameter[Int, 1, 2]()


# Variadic list initialization of List does not work in alias domain
# https://github.com/modularml/modular/issues/33579

# CHECK-LABEL: lit.fn @"init_self_memory_variadics
fn init_self_memory_variadics():
    # 1 and 2 need to be passed through memory in the variadics.
    # CHECK-NEXT: lit.alias.decl *"x`":
    # CHECK-SAME:  [store_to_mem({1}), store_to_mem({2})]
    alias x = MyList[Int](1, 2)

struct MyList[T: Copyable]:
    @implicit
    fn __init__(out self, *values: T): pass

# Infer-only parameters should be bindable with keywords
alias ImmMyStringSlice = MyStringSlice[is_mutable=False]
struct MyStringSlice[is_mutable: Bool, //, origin: Origin[is_mutable]]:  pass

# This only binds to immutable things.
# CHECK-LABEL: lit.fn @"test_imm_string_slice
# CHECK-SAME: (%a: !lit.ref<{{.*}}@MyStringSlice<:!Bool {:i1 0},
fn test_imm_string_slice(a: ImmMyStringSlice):
    pass





##===----------------------------------------------------------------------===##
# Function Overloading on Parameters
##===----------------------------------------------------------------------===##


fn parameter_overloading[param: Int]():
    pass

fn parameter_overloading[param: DType]():
    pass

fn partial_parameter_overloading[param: Int, other: Int]():
    pass

fn partial_parameter_overloading[param: DType, other: DType]():
    pass

# CHECK-LABEL: lit.fn @"form_reference_to_overloaded
fn form_reference_to_overloaded():
    # CHECK-NEXT: @"parameter_overloading[[[INT:.*Int]]]()"<:!Int {1}>
    alias refresult = parameter_overloading[1]
    # CHECK-NEXT: !lit.generator<<"other": !Int>() -> !kgen.none> = <{{.*}}@"partial_parameter_overloading[[[INT]],[[INT]]]()"<:!Int {1}, :!Int ?>
    alias partial = partial_parameter_overloading[1]

##===----------------------------------------------------------------------===##
# Parameter Inference
##===----------------------------------------------------------------------===##

@register_passable("trivial")
struct StaticVec[size: Int]:
  fn __init__[type: __mlir_type.`!kgen.dtype`](out self, v: __mlir_type[`!pop.simd<`, size.value, `, `, type, `>`]):
      pass

  @staticmethod
  fn thing[type: __mlir_type.`!kgen.dtype`](v: __mlir_type[`!pop.simd<`, size.value, `, `, type, `>`]):
      return

fn callee1[size: Int](v: StaticVec[size]): pass
fn callee2[T: __mlir_type.`!kgen.type`](v: T): pass
fn callee3[size: Index, type: __mlir_type.`!kgen.dtype`]
   (v:  __mlir_type[`!pop.simd<`, size, `, `, type, `>`]): pass
fn callee4[T: __mlir_type.`!kgen.type`]
   (v:  __mlir_type[`!kgen.pointer<`, T, `>`]): pass

# CHECK-LABEL: lit.fn @"testParamInference{{.*}}"<size: !Int>(
fn testParamInference[size: Int](a: StaticVec[4], b: StaticVec[size],
                                 b2: StaticVec[size+2],
                                 c: __mlir_type.`!pop.simd<17, f32>`,
                                 d: __mlir_type.`!kgen.pointer<f32>`):
  # CHECK-NEXT: lit.call @{{.*}}callee1{{.*}}<{{.*}}4{{.*}}>(%a)
  callee1(a)
  # CHECK-NEXT: lit.call @{{.*}}callee1{{.*}}<:!Int size>(%b)
  callee1(b)
  # CHECK-NEXT: lit.call @{{.*}}callee1{{.*}}<:!Int {value = add(#lit.struct.extract<:!Int size, "value">, 2)}>(%b2)
  callee1(b2)
  # CHECK-NEXT: lit.call @{{.*}}callee2{{.*}}<:type @parameters::@StaticVec<:!Int size>{{.*}}>(%b)
  callee2(b)
  # CHECK-NEXT: lit.call @{{.*}}callee3{{.*}}<17, :dtype f32>(%c)
  callee3(c)
  # CHECK-NEXT: lit.call @{{.*}}callee4{{.*}}<:type f32>(%d)
  callee4(d)

# CHECK-LABEL: lit.struct.decl @Abstraction
# CHECK-SAMEL <[[A:.*]]: !Int>
@value
@register_passable("trivial")
struct Abstraction[a: Int]:
  alias val = a.value

  @implicit
  fn __init__(out self, arg: Int):
    pass

  @staticmethod
  fn push[b: Int]() -> Abstraction[a + b]:
      return Abstraction[a + b]()

  @staticmethod
  fn pull[b: Int](value: Abstraction[a + b]):
      return

# CHECK-LABEL: lit.fn @"testDependentType{{.*}}"<
# CHECK-SAME: rank: !Int, shape: array<#lit.struct.extract<:!Int rank, "value">
fn testDependentType[
    rank: Int,
    shape: __mlir_type[`!pop.array<`, rank.value, `, index>`],
]():
    pass

@no_inline
fn dont_interpret():
  pass

# CHECK-LABEL: lit.fn @"testParameterEvaluator()"
fn testParameterEvaluator():
  # CHECK-NEXT: lit.alias.decl *"x{{.*}}" = <1>
  alias x = Abstraction[1].val
  # CHECK-NEXT: %y = lit.var.decl "y"
  # CHECK-NEXT: %0 = lit.call @parameters::@Abstraction::@"push{{.*}}"<:!Int {1}, :!Int {2}>
  # CHECK-NEXT: lit.ref.store %0, %y
  var y : Abstraction[3] = Abstraction[1].push[2]()
  # CHECK-NEXT: [[Y:%.*]] = lit.ref.load %y : {{.*}}@Abstraction<:!Int {3}>,
  # CHECK-NEXT: lit.call {{.*}}@Abstraction::@"pull{{.*}}"<{{.*}}>([[Y]])
  Abstraction[1].pull[2](y)
  # CHECK-NEXT: lit.call {{.*}}@"testDependentType{{.*}}"<:!Int {1}, :array<1, index> [0]>
  testDependentType[1, __mlir_attr.`#pop.array<0> : !pop.array<1, index>`]()

  # CHECK: lit.call {{.*}}dont_interpret
  dont_interpret()


fn takeAbstraction2(value: Abstraction[2]):
    return

@register_passable
struct AnotherAbstraction[a: Int]:
    var value : Abstraction[a + 1]

    fn __init__(out self):
        self.value = Abstraction[a + 1]()

    fn __copyinit__(out self, existing: Self):
        self.value = existing.value

# CHECK-LABEL: lit.fn @"testDependentField()"
fn testDependentField():
    var lvalue = AnotherAbstraction[1]()
    # CHECK: [[VALUE_PTR:%.*]] = lit.ref.struct.ger %lvalue[value] {{.*}}@AnotherAbstraction<:!Int {1}>,{{.*}}@Abstraction<:!Int {2}>
    takeAbstraction2(lvalue.value)

struct LeafToRootEval[a: Int, b: Int]:
    var value: Abstraction[a + b + a]

# CHECK-LABEL: lit.fn @"refine_type_leaf_to_root
fn refine_type_leaf_to_root(e: LeafToRootEval[2, 3]):
    # CHECK: lit.var.decl "value" {{.*}}@Abstraction<:!Int {7}>
    var value = e.value

fn tail_types[T: AnyTrivialRegType, *U: AnyType](a: T, *b: *U):
    pass

# CHECK-LABEL: lit.fn @"call_with_tail_types()"
fn call_with_tail_types():
    # CHECK: call {{.*}}tail_types{{.*}}<:type !Int, :variadic<!AnyType> []>
    tail_types(1)
    # CHECK: call {{.*}}tail_types{{.*}}<:type !Int, :variadic<!AnyType> [{{\[}}!FloatDyn, {{.*}}]]>
    tail_types(1, 1.2)
    # CHECK: call {{.*}}tail_types{{.*}}<:type !Int, :variadic<!AnyType> [{{\[}}!Int, {{.*}}]]>
    tail_types(1, 77)

# COM: We can't infer parameters from the default value, but we need to test if
# COM: if other parameters are inferred correctly in their presence.
fn infer_with_default_arg[T: AnyTrivialRegType](a: T, b: Int = 7):
    pass

# CHECK-LABEL: lit.fn @"test_infer_with_default_arg()"
fn test_infer_with_default_arg():
    # lit.call @{{.*}}::@"infer_with_default_arg[AnyTrivialRegType]($0,::Int)"<:type !Int>
    infer_with_default_arg(128)

# CHECK-LABEL: lit.fn @"indirect_call_infer_params
fn indirect_call_infer_params[callee: fn[x: Int](y: Abstraction[x])->None]():
    # CHECK: call[!lit.generator<("y": {{.*}}#Abstraction <:!Int {2}>
    # CHECK-SAME: bind_params(:!lit.generator<<"x": !Int>("y": {{.*}}Abstraction <:!Int *(0,0)>
    # CHECK-SAME: callee, {2}
    callee(Abstraction[2]())

# COM: test parameter inference through signatureType,
# COM: from issue https://github.com/modular/mojo/issues/1362
fn mapSingle[A: AnyType, B: AnyType, R: AnyType](
  f: fn(x: A, y: B) -> R,
  a: A, b: B
) -> R:
  return f(a, b)
fn useMapSingle() -> String:
  fn f(x: String, y: String) -> String:
    return String()
  # CHECK: lit.call {{.*}}mapSingle{{.*}}<:!AnyType [!String, {{.*}}], :!AnyType [!String, {{.*}}], :!AnyType [!String, {{.*}}]>
  return mapSingle(f, "a", "b")


# COM: Test that keyword-only parameter can be inferred after variadic.
# COM: Issue https://github.com/modularml/modular/issues/33939
fn deduce_kw_only[*Ts: Int, x: Int](y: Abstraction[x]):
    pass


# CHECK-LABEL: lit.fn @"out_of_order_kw
fn out_of_order_kw[x: Index, y: IndexParam[x]]():
    # CHECK-NEXT: out_of_order_kw{{.*}}<0, :{{.*}}IndexParam<0> {{.*}}IndexParam::@"__init__{{.*}}<0>, #kgen.none)>>
    alias bound = out_of_order_kw[y=None, x=`0`]


# CHECK-LABEL: lit.fn @"test_deduce_kw_only
fn test_deduce_kw_only(a: Abstraction[3]):
    # CHECK: call {{.*}}@"deduce_kw_only{{.*}}<:variadic<!Int> [{1}, {2}], :!Int {3}>(%a)
    deduce_kw_only[1, 2](a)

# Make sure the +1 in the 'a' argument doesn't break inference.
fn test_infer_add(a: SIMD[DType.float32, 4], b: SIMD[DType.int32, 5]):
   _ = take_two(a, b)

struct CallableArg[ArgT: AnyTrivialRegType]:
    fn __call__(self, arg: ArgT):
        pass

# CHECK-LABEL: lit.fn @"infer_conversion_arg_type
fn infer_conversion_arg_type(callable: CallableArg[NoneType]):
    # CHECK: lit.call {{.*}}CallableArg::@"__call__{{.*}}<:type !NoneType>
    callable(None)

fn take_two[a_type: DType, c_type: DType, width: Int](
    c: SIMD[c_type, width], a: SIMD[a_type, width + 1],
) -> SIMD[c_type, width]: pass

fn implicit_signature[
    type: DType,
    rank: Int, //,
    func: fn[width: Int](Abstraction[rank]) -> SIMD[type, width],
]():
    pass

# CHECK-LABEL: lit.fn @"signature_inference
fn signature_inference[dt: DType, rank: Int]():
    fn func[width: Int](idx: Abstraction[rank]) -> SIMD[dt, width]:
        pass

    # CHECK: call {{.*}}implicit_signature{{.*}}<:!DType dt, :!Int rank,
    # CHECK-SAME: :!lit.generator<<"width": !Int>(!lit.struct<#Abstraction <:!Int rank>
    # CHECK-SAME: -> !lit.struct<#SIMD <:!DType dt, :!Int *(0,0)>>
    implicit_signature[func]()


struct ClosureParam[lt: MutableOrigin, f: fn () capturing [lt] -> None]:
    fn __moveinit__(out self, owned existing: Self):
        pass


# CHECK-LABEL: lit.fn @"infer_implicit_params
fn infer_implicit_params(owned p: ClosureParam):
    # CHECK: call {{.*}}ClosureParam::@"__moveinit__{{.*}}<:origin<1> *"lt`", :!lit.generator<:{mut *"lt`"}:() capturing -> !kgen.none> *"f`1">
    var tmp = p^
    _ = tmp^


trait ToInt:
    fn to_int(self) -> Int:
        pass

@value
@register_passable("trivial")
struct HasToInt(ToInt):
    var inner: Int
    @always_inline("nodebug")
    fn to_int(self) -> Int:
        return self.inner

# COM: https://linear.app/modularml/issue/MOCO-885/crash-when-using-autoparam-in-parametrized-structs
@value
@register_passable("trivial")
struct MixedInferAndPosParam[size: Int]:
    var f0: Int

    # CHECK-LABEL: lit.fn @"__init__[{{.*}}ToInt](
    # CHECK-SAME: T0: !ToInt, T1: !ToInt
    fn __init__[T0: ToInt, T1: ToInt, //](out self, a: T0, b: T1):
        self.f0 = a.to_int()

@value
@register_passable("trivial")
struct MixedInferAndPosParamWithInferredOnStruct[ST: ToInt, //, size: Int]:
    var f0: Int

    # CHECK-LABEL: lit.fn @"__init__[{{.*}}ToInt](
    # CHECK-SAME: T0: !ToInt, T1: !ToInt
    fn __init__[T0: ToInt, T1: ToInt, //](out self, z: ST, a: T0, b: T1):
        self.f0 = a.to_int()

# CHECK-LABEL: lit.fn @"useMixedInferAndPosParam()"
fn useMixedInferAndPosParam():
    # CHECK: lit.call {{.*}}::@MixedInferAndPosParam::@"__init__{{.*}}<:!Int {27}, :!ToInt [!HasToInt, {{.*}}], :!ToInt [!HasToInt, {{.*}}]
    _ = MixedInferAndPosParam[27](HasToInt(37), HasToInt(47))
    # CHECK: lit.call {{.*}}::@MixedInferAndPosParamWithInferredOnStruct::@"__init__{{.*}}<:!ToInt [!HasToInt, {{.*}}], :!Int {27}, :!ToInt [!HasToInt, {{.*}}], :!ToInt [!HasToInt, {{.*}}]
    _ = MixedInferAndPosParamWithInferredOnStruct[27](HasToInt(99), HasToInt(37), HasToInt(47))

@register_passable("trivial")
struct Box[T: AnyType]:
    @implicit
    fn __init__(out self, x: T):
        pass

# CHECK-LABEL: lit.fn @"infer_box_type
fn infer_box_type[T: AnyType, //, box: Box[T]]():
    # CHECK-NEXT: lit.call {{.*}}infer_box_type{{.*}}<:!AnyType [!Int,
    infer_box_type[Int()]()

# MOCO-1457: Support struct param inference for origins
struct OriginStructInferenceImm[origin: Origin[False]]:
    fn __init__(out self, ref [origin._mlir_origin]data: Int):  pass
struct OriginStructInferencePar[is_mutable: Bool, //, origin: Origin[is_mutable]]:
    fn __init__(out self, ref [origin._mlir_origin]data: Int):  pass
struct OriginStructInferenceParWrapped[is_mutable: Bool, //, origin: Origin[is_mutable]]:
    fn __init__(out self, ref [origin]data: Int):  pass
struct OriginStructInferenceParSpecialized[is_mutable: Bool, //, origin: Origin[is_mutable]]:
    fn __init__[O: Origin[False]](out self: OriginStructInferenceParSpecialized[O], ref [O]data: Int):  pass

# CHECK-LABEL: lit.fn @"test_origin_struct_inf
fn test_origin_struct_inf[imm_data: Int](mut data: Int):
   # This needs to infer the origin through an implicit conversion
   # CHECK: %0 = lit.ref.immut %data
   # CHECK-NEXT: lit.call {{.*}}OriginStructInferenceImm::@"__init__
   # CHECK-SAME: {_mlir_origin: origin<0> = (mutcast mut *"data`")}>(%0, %immTest)
   immTest = OriginStructInferenceImm(data)

   # CHECK-NEXT: lit.call {{.*}}OriginStructInferencePar::@"__init__
   # CHECK-SAME: {_mlir_origin: origin<1> = *"data`"}>(%data, %parTest)
   parTest = OriginStructInferencePar(data)

   # CHECK-NEXT: lit.call {{.*}}OriginStructInferenceParWrapped::@"__init__
   # CHECK-SAME: {_mlir_origin: origin<1> = *"data`"}>(%data, %parWrappedTest)
   parWrappedTest = OriginStructInferenceParWrapped(data)

   # CHECK: %[[IMMUT:.+]] = lit.ref.immut {{.*}} : <!Int, mut [[IMMUT_REF:.+]]>
   # CHECK-NEXT: lit.call {{.*}}OriginStructInferenceParSpecialized::@"__init__
   # CHECK-SAME: :!Bool {:i1 0},
   # CHECK-SAME: :{{[^ ]*}}Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = (mutcast mut [[IMMUT_REF]])},
   # CHECK-SAME: :{{[^ ]*}}Origin<:!Bool {:i1 0}> {_mlir_origin: origin<0> = (mutcast mut [[IMMUT_REF]])}>
   # CHECK-SAME: (%[[IMMUT]], %parSpecializedTest)
   parSpecializedTest = OriginStructInferenceParSpecialized(imm_data)


##===----------------------------------------------------------------------===##
# Access parameter through structure
##===----------------------------------------------------------------------===##

struct MultiStruct[p1: Int, p2: Int, p3: Int]:
    fn __init__(out self): pass

fn foo[x: Int]():
  pass

fn bar(x : Int):
  pass

# CHECK-LABEL: lit.fn @"reference_params_through_struct
fn reference_params_through_struct():
    var x = MultiStruct[52, 9, 33]()

    # CHECK: %[[Y:.*]] = lit.var.decl "y"
    # CHECK-NEXT: %[[P:.*]] = kgen.param.constant: {{.*}} <{52}
    # CHECK-NEXT: lit.ref.store %[[P]], %[[Y]]
    var y = x.p1

    # CHECK: %[[P:.*]] = kgen.param.constant: {{.*}} <{9}
    # CHECK-NEXT: lit.call @{{.*}}bar({{.*}})"(%[[P]])
    bar(x.p2)

    # CHECK: lit.call @{{.*}}foo{{.*}}<:!Int {33}>
    foo[x.p3]()

    # CHECK: %[[Z:.*]] = lit.var.decl "z"
    # CHECK-NEXT: %[[P:.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: lit.ref.store %[[P]], %[[Z]]
    var z = MultiStruct[1, 2, 3].p1

    # CHECK: %[[P:.*]] = kgen.param.constant: !Int = <{2}>
    # CHECK-NEXT: lit.call @{{.*}}bar({{.*}})"(%[[P]])
    bar(MultiStruct[1, 2, 3].p2)

    # CHECK: lit.call @{{.*}}foo{{.*}}<:!Int {3}>
    foo[MultiStruct[1, 2, 3].p3]()


@register_passable
struct DependentParam[x: Index, y: ParamType[x]]:
    pass


# CHECK-LABEL: lit.fn @"auto_param_dependent
# CHECK-SAME: <?, [[Y0:.*]], [[Y1:.*]]: {{.*}}ParamType<[[Y0]]>>
fn auto_param_dependent(value: DependentParam[*_]):
    # CHECK-NEXT: ParamType<[[Y0]]> = <[[Y1]]>
    alias param = value.y


##===----------------------------------------------------------------------===##
# Default function parameters
##===----------------------------------------------------------------------===##

fn default_params[a: Int, b: Int = 7, c: String = "woof"]():
    pass


# CHECK-LABEL: lit.fn @"test_default_params()"
fn test_default_params():
    # CHECK: lit.call @{{.*}}@"default_params[::Int,::Int,::String]()"
    # CHECK-SAME: <:!Int {1}, :!Int {7}, {{.*}}#StringLiteral <:string "woof">
    default_params[1]()

    # CHECK: lit.call @{{.*}}@"default_params[::Int,::Int,::String]()"
    # CHECK-SAME: <:!Int {2}, :!Int {8}, {{.*}}#StringLiteral <:string "woof">
    default_params[2, 8]()

    # CHECK: lit.call @{{.*}}@"default_params[::Int,::Int,::String]()"
    # CHECK-SAME: <:!Int {4}, :!Int {9}, {{.*}}#StringLiteral <:string "meow">
    default_params[4, 9, "meow"]()


fn test_indirect_default_params[
    callee: fn[a: Int, b: Int = 7, c: String = "woof"]()->None]():

    # CHECK: lit.call[!lit.generator<() -> !kgen.none>: bind_params(:!lit.generator<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> callee,
    # CHECK-SAME: {1}, {7}, {{.*}}#StringLiteral <:string "woof"
    callee[1]()

    # CHECK: lit.call[!lit.generator<() -> !kgen.none>: bind_params(:!lit.generator<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> callee,
    # CHECK-SAME: {2}, {8}, {{.*}}#StringLiteral <:string "woof"
    callee[2, 8]()

    # CHECK: lit.call[!lit.generator<() -> !kgen.none>: bind_params(:!lit.generator<<"a": {{.*}}, "b": {{.*}}, "c": {{.*}}>() -> !kgen.none> callee,
    # CHECK-SAME: {4}, {9}, {{.*}}#StringLiteral <:string "meow"
    callee[4, 9, "meow"]()


# COM: check that inferred parameter values take precedence over defaults
# CHECK-LABEL: lit.fn @"inferred_default_param
fn inferred_default_param[dt: DType, w: Int = 8](a: SIMD[dt, w]):
    pass


# CHECK: lit.fn @"test_inferred_default_param{{.*}}"<x: !Int>
# CHECK: lit.call @{{.*}}@"inferred_default_param{{.*}}"<:!DType {:dtype f32}, :!Int {4}>
# CHECK: lit.call @{{.*}}@"inferred_default_param{{.*}}"<:!DType {:dtype f32}, :!Int x>
fn test_inferred_default_param[
    x: Int
](concrete: SIMD[DType.float32, 4], p: SIMD[DType.float32, x]):
    inferred_default_param(concrete)
    inferred_default_param(p)


# COM: basic check for memory-only default parameters
@value
struct MemoryOnlyType:
    pass


# CHECK: lit.fn @"mem_only_default_param[{{.*}}MemoryOnlyType::@"__init__()
fn mem_only_default_param[x: MemoryOnlyType = MemoryOnlyType()]():
    pass

# CHECK-LABEL: lit.fn @"test_mem_only_default_param()"
# CHECK: lit.call @{{.*}}@"mem_only_default_param[{{.*}}MemoryOnlyType::@"__init__()
fn test_mem_only_default_param():
    mem_only_default_param()

# CHECK-LABEL: lit.fn @"param_default{{.*}}"<
# CHECK-SAME: x: !Int = {1}>(%y: !Int = x)
fn param_default[x: Int = 1](y: Int = x):
    pass

# CHECK-LABEL: lit.fn @"test_param_default
fn test_param_default():
    # CHECK: [[C:%.*]] = kgen.param.constant: !Int = <{4}>
    # CHECK-NEXT: call {{.*}}param_default{{.*}}<:!Int {4}>([[C]]
    param_default[4]()
    # CHECK: [[C:%.*]] = kgen.param.constant: !Int = <{1}>
    # CHECK-NEXT: call {{.*}}param_default{{.*}}<:!Int {1}>([[C]]
    param_default()

struct Optional[T: AnyType]:
    @implicit
    fn __init__(out self, none: __mlir_type.`!kgen.none`):
        pass

    @implicit
    fn __init__(out self, value: T):
        pass

fn default_on_infer_failure[p: Index = `0`](a: Optional[ParamType[p]] = None):
    pass

# CHECK-LABEL: lit.fn @"test_optional_inference
fn test_optional_inference(value: ParamType[`3`]):
    # CHECK-NEXT: [[NONE:%.*]] = lit.var.decl {{.*}}ParamType<0>
    # CHECK: [[IMMUT:%.*]] = lit.ref.immut [[NONE]]
    # CHECK-NEXT: call {{.*}}default_on_infer_failure{{.*}}<0>([[IMMUT]])
    default_on_infer_failure()

    # CHECK: call {{.*}}default_on_infer_failure{{.*}}<0>
    default_on_infer_failure(None)

    # CHECK: call {{.*}}default_on_infer_failure{{.*}}<3>
    default_on_infer_failure(value)

##===----------------------------------------------------------------------===##
# Default struct parameters
##===----------------------------------------------------------------------===##

# CHECK: lit.struct.decl @DefaultParams<{{.*}}: !Int, {{.*}}: !Int = {7}, {{.*}}: {{.*}}#StringLiteral <:string "woof">
@value
struct DefaultParams[a: Int, b: Int = 7, msg: String = "woof"]: pass

# CHECK-LABEL: lit.fn @"test_default_param_struct()"
fn test_default_param_struct():
    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int {1}, :!Int {7}, {{.*}}#StringLiteral <:string "woof">
    alias T = DefaultParams[1]
    # CHECK-NEXT: %[[INIT:.*]] = lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@DefaultParams<
    # CHECK-SAME:   :!Int {1}, :!Int {7}, {{.*}}#StringLiteral <:string "woof">
    # CHECK-NEXT: lit.call @{{.*}}@DefaultParams::@"__init__({{.*}}<:!Int {1}, :!Int {7}, {{.*}}#StringLiteral <:string "woof">
    _ = DefaultParams[1]()

    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int {2}, :!Int {3}, {{.*}}#StringLiteral <:string "woof">
    alias U = DefaultParams[2, 3]
    # CHECK-NEXT: %[[INIT:.*]] = lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@DefaultParams<
    # CHECK-SAME:   :!Int {2}, :!Int {3}, {{.*}}#StringLiteral <:string "woof">
    # CHECK-NEXT: lit.call @{{.*}}@DefaultParams::@"__init__({{.*}}<:!Int {2}, :!Int {3}, {{.*}}#StringLiteral <:string "woof">
    _ = DefaultParams[2, 3]()

    # CHECK: lit.alias.decl {{.*}}@DefaultParams<
    # CHECK-SAME: :!Int {4}, :!Int {5}, {{.*}}#StringLiteral <:string "meow">
    alias S = DefaultParams[4, 5, "meow"]
    # CHECK-NEXT: %[[INIT:.*]] = lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@DefaultParams<
    # CHECK-SAME:   :!Int {4}, :!Int {5}, {{.*}}#StringLiteral <:string "meow">
    # CHECK-NEXT: lit.call @{{.*}}@DefaultParams::@"__init__({{.*}}<:!Int {4}, :!Int {5}, {{.*}}#StringLiteral <:string "meow">
    _ = DefaultParams[4, 5, "meow"]()


# CHECK: lit.struct.decl @AllDefaultParams<{{.*}}: !Int = {0}, {{.*}}MemoryOnlyType::@"__init__()
@value
struct AllDefaultParams[x: Int = 0, v: MemoryOnlyType = MemoryOnlyType()]: pass

# CHECK-LABEL: lit.fn @"test_default_param_struct_all_default()"
fn test_default_param_struct_all_default():
    # CHECK: lit.alias.decl *"T{{.*}}": meta<!lit.struct<{{.*}}#AllDefaultParams{{.*}}>> = <@{{.*}}::@AllDefaultParams<
    # CHECK-SAME: :!Int {0},
    # CHECK-SAME: :!MemoryOnlyType {{.*}}MemoryOnlyType::@"__init__()
    alias T = AllDefaultParams[]

    # CHECK: %[[INIT:.*]] = lit.var.decl {{.*}} : !lit.ref<@{{.*}}::@AllDefaultParams<
    # CHECK-SAME:   :!Int {0}, :!MemoryOnlyType {{.*}}MemoryOnlyType::@"__init__()
    # CHECK-NEXT: = lit.call @{{.*}}::@AllDefaultParams::@"__init__({{.*}}<:!Int {0}, :!MemoryOnlyType
    _ = AllDefaultParams[]()


# COM: Issue #22763
fn IntForType[T: AnyTrivialRegType]() -> Int:
    return 1

struct StructWithParametricDefaultValue[T: AnyTrivialRegType, N: Int = IntForType[T]()]:
    pass

# CHECK-LABEL: lit.fn @"test_struct_with_parametric_default_value()"
fn test_struct_with_parametric_default_value():
    # CHECK: lit.alias.decl *"a{{.*}}": meta<!lit.struct<{{.*}}>> = <@{{.*}}::@StructWithParametricDefaultValue<
    # CHECK-SAME: :type !Int
    # CHECK-SAME: :!Int apply(:!lit.generator<() -> !Int> @{{.*}}::@"IntForType[AnyTrivialRegType]()"{{.*}}<:type !Int>)>
    alias a = StructWithParametricDefaultValue[Int]

##===----------------------------------------------------------------------===##
# Struct keyword parameters
##===----------------------------------------------------------------------===##

@value
struct KwParamStruct[a: Int, b: Int = 2, c: Int = 3]: pass

# CHECK-LABEL: lit.fn @"test_struct_kw_params()"
fn test_struct_kw_params():
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int {5}, :!Int {7}, :!Int {3}
    _ = KwParamStruct[5, b=7]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int {5}, :!Int {7}, :!Int {9}
    _ = KwParamStruct[5, b=7, c=9]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int {5}, :!Int {2}, :!Int {9}
    _ = KwParamStruct[5, c=9]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int {5}, :!Int {7}, :!Int {9}
    _ = KwParamStruct[5, c=9, b=7]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int {5}, :!Int {7}, :!Int {9}
    _ = KwParamStruct[a=5, c=9, b=7]()
    # CHECK: lit.var.decl {{.*}} synth : !lit.ref<@{{.*}}::@KwParamStruct<:!Int {5}, :!Int {7}, :!Int {9}
    _ = KwParamStruct[c=9, b=7, a=5]()

##===----------------------------------------------------------------------===##
# Partial binding
##===----------------------------------------------------------------------===##

@value
struct Thing[v: Int]: pass

struct CtadStruct[a: Int, b: Int]:
    @implicit
    fn __init__(out self, x: Thing[a]): pass

    fn __init__(out self, x: Thing[a], y: Thing[b]): pass

    @staticmethod
    fn foo(x: Thing[a]): pass

    @staticmethod
    fn foo(x: Thing[a], y: Thing[b]): pass

struct CtadStructWithDefault[a: Int, b: Int, c: Int = 8]:
    @implicit
    fn __init__(out self, x: Thing[a]): pass

    fn __init__(out self, x: Thing[a], y: Thing[b]): pass

    @staticmethod
    fn foo(x: Thing[a]): pass

    @staticmethod
    fn foo(x: Thing[a], y: Thing[b]): pass


struct CtadStructWithMultiDefault[a: Int, b: Int = 6, c: Int = 8, d: Int = 10]:
    @implicit
    fn __init__(out self, x: CtadStructWithMultiDefault[a]): pass


# CHECK-LABEL: lit.fn @"test_partial_binding_CTAD(
fn test_partial_binding_CTAD(multi: CtadStructWithMultiDefault[5]):
    # CHECK: call @{{.*}}::@CtadStruct::@"__init__({{.*}})"{{.*}}<:!Int {6}, :!Int {7}>
    _ = CtadStruct[b=7](Thing[6]())
    # CHECK: call @{{.*}}::@CtadStruct::@"__init__({{.*}})"{{.*}}<:!Int {8}, :!Int {9}>
    _ = CtadStruct[](Thing[8](), Thing[9]())
    # CHECK: call @{{.*}}::@CtadStruct::@"foo({{.*}}<:!Int {6}, :!Int {7}>
    CtadStruct[b=7].foo(Thing[6]())
    # CHECK: call @{{.*}}::@CtadStruct::@"foo({{.*}}<:!Int {8}, :!Int {9}>
    CtadStruct[].foo(Thing[8](), Thing[9]())

    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int {6}, :!Int {7}, :!Int {8}>
    _ = CtadStructWithDefault[b=7](Thing[6]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int {2}, :!Int {1}, :!Int {8}>
    _ = CtadStructWithDefault[](y=Thing[1](), x=Thing[2]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"__init__({{.*}})"{{.*}}<:!Int {6}, :!Int {9}, :!Int {8}>
    _ = CtadStructWithDefault(Thing[6](), Thing[9]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int {6}, :!Int {7}, :!Int {8}>
    CtadStructWithDefault[b=7].foo(Thing[6]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int {2}, :!Int {1}, :!Int {8}>
    CtadStructWithDefault[].foo(y=Thing[1](), x=Thing[2]())
    # CHECK: call @{{.*}}::@CtadStructWithDefault::@"foo({{.*}}<:!Int {4}, :!Int {3}, :!Int {8}>
    CtadStructWithDefault.foo(y=Thing[3](), x=Thing[4]())

    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {5}, :!Int {6}, :!Int {9}, :!Int {10}>
    _ = CtadStructWithMultiDefault[_, _, 9, _](multi)
    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {5}, :!Int {6}, :!Int {8}, :!Int {9}>
    _ = CtadStructWithMultiDefault[_, _, _, 9](multi)
    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {5}, :!Int {3}, :!Int {8}, :!Int {9}>
    _ = CtadStructWithMultiDefault[_, 3, _, 9](multi)
    # CHECK: call @{{.*}}::@CtadStructWithMultiDefault::@"__init__({{.*}}<:!Int {5}, :!Int {6}, :!Int {8}, :!Int {10}>
    _ = CtadStructWithMultiDefault[5, _, _, _](multi)


# COM: https://github.com/modular/mojo/issues/1227
# COM: Ensure default parameters are rebound during CTAD.
@value
@register_passable("trivial")
struct DependentDefault[x: Int = 1, y: Int = x]:
    pass


# CHECK-LABEL: lit.fn @"dependent_default_ctad
fn dependent_default_ctad():
    # CHECK-NEXT: value{{.*}}: {{.*}}@DependentDefault<:!Int {1}, :!Int {1}>
    alias value = DependentDefault()


alias Scalar = SIMD[_, 1]


# CHECK-LABEL: lit.fn @"scalar_type{{.*}}"<dt: !DType>
fn scalar_type[dt: DType]():
    # CHECK: alias.decl [[T:.*]]: meta<{{.*}}SIMD<:!DType dt, :!Int {1}>>
    alias T = Scalar[dt]

    #FIXME(29495): reenable.
    # https://github.com/modularml/modular/issues/29495
    # HECK: lit.var.decl "value" = %{{.*}} : !lit.struct<{{.*}}@SIMD<:!DType dt,
    #var value: T = 1
    # HECK: call {{.*}}<:!DType dt, {{.*}}, :!DType dt>(%value)
    #_ = value.cast[dt]()

# CHECK-LABEL: lit.fn @"funct_partial_binding{{.*}}"<x: !Empty, F:
fn funct_partial_binding[x: Empty, F: fn[t: Empty, s: Empty] () -> None]():
    # CHECK: !lit.generator<<"u": !Empty, "v": !Empty>() -> !kgen.none> = <rebind(
    # CHECK-SAME: :!lit.generator<<"t": !Empty, "s": !Empty>() -> !kgen.none>
    # CHECK-SAME: bind_params(:!lit.generator<<"t": !Empty, "s": !Empty>() -> !kgen.none> F, ?, ?)

    alias G: fn[u: Empty, v: Empty] () -> None = F[s=_, t=_]
    # CHECK: !lit.generator<<"u": !Empty>() -> !kgen.none> = <rebind(
    # CHECK-SAME: :!lit.generator<<"s": !Empty>() -> !kgen.none>
    # CHECK-SAME: bind_params(:!lit.generator<<"t": !Empty, "s": !Empty>() -> !kgen.none> F, x, ?))>
    alias H: fn[u: Empty] () -> None = F[x]

@value
struct StructWithSpecificSelfInitTypes[size: Int]:
    fn __init__(out self: StructWithSpecificSelfInitTypes[0]): pass
    @implicit
    fn __init__(out self: StructWithSpecificSelfInitTypes[1], a: Int): pass
    fn __init__(out self: StructWithSpecificSelfInitTypes[2], a: Int, b: Int): pass

struct DependentSpecificInitSelf[T: AnyType]:
    @implicit
    fn __init__[U: Movable](out self: DependentSpecificInitSelf[U], owned value: U):
        pass

fn implicit_convert_specific_Self(value: StructWithSpecificSelfInitTypes[1]):
    pass

# CHECK-LABEL: lit.fn @"test_inference_from_Self_type
fn test_inference_from_Self_type(x: Int):
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {0}>([[TMP]])
  _ = StructWithSpecificSelfInitTypes()
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {1}>(%x, [[TMP]])
  _ = StructWithSpecificSelfInitTypes(x)
  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {2}>(%x, %x, [[TMP]])
  _ = StructWithSpecificSelfInitTypes(x, x)

  # CHECK-NEXT: [[TMP:%.*]] = lit.var.decl "anonymous
  # CHECK-NEXT: lit.call {{.*}}__init__{{.*}}<:!Int {1}>(%x, [[TMP]])
  # CHECK-NEXT: [[IMM:%.*]] = lit.ref.immut [[TMP]]
  # CHECK-NEXT: call {{.*}}implicit_convert_specific_Self{{.*}}([[IMM]])
  implicit_convert_specific_Self(x)

  # CHECK: [[TMP:%.*]] = lit.var.decl "anonymous
  # CHECK: lit.call {{.*}}__init__{{.*}}<:!AnyType [!Int, {{.*}}], :!Movable [!Int, {{.*}}]>{{.*}}([[TMP]], {{.*}})
  _ = DependentSpecificInitSelf(x)

struct AutoParamDefault[value: Index, param: Index, default: Index = param]:
    @implicit
    fn __init__(out self, ptr: ParamType[value]): pass
    fn __init__(out self, *, other: Self): pass
    fn method(self, other: ParamType[value]): pass
    fn method(self, other: AutoParamDefault[value, *_]): pass

# CHECK-LABEL: lit.fn @"implicit_conversion_overload
fn implicit_conversion_overload(x: AutoParamDefault[`1`], ptr: ParamType[`1`]):
    # CHECK: call {{.*}}method{{.*}}(%x, %ptr)
    x.method(ptr)

# MOCO-1144
# https://linear.app/modularml/issue/MOCO-1144/[mojo-lang]-crash-on-partially-bound-parameter-list
fn takeAnyTypeReturnInt[t: AnyType]() -> Int: pass
struct MOCO1144[
    is_mutable: Bool,
    type: AnyType,
    alignment: Int = takeAnyTypeReturnInt[type]()
]: pass
alias MOCO1144Bound = MOCO1144[True, _, _]

fn getMOCO1144Bound() -> MOCO1144Bound[Int]: pass

# CHECK-LABEL: lit.fn @"tryCallingAThingReturningMOCO1144Bound
fn tryCallingAThingReturningMOCO1144Bound():
    # CHECK-NEXT:  lit.var.decl "x" {{.*}}MOCO1144<:!Bool {:i1 1}, :!AnyType [!Int{{.*}}takeAnyTypeReturnInt[::AnyType]()"<:!AnyType [!Int
    var x = getMOCO1144Bound()


##===----------------------------------------------------------------------===##
# Origin Parameters
##===----------------------------------------------------------------------===##


@register_passable("trivial")
struct SomeReference[lt: __mlir_type.`!lit.origin<0>`]:
    pass


# CHECK-LABEL: lit.fn @"unbound_origin
# CHECK-SAME: <?, [[R:.*]]: origin<0>>
# CHECK-SAME: #SomeReference <:origin<0> [[R]]>
fn unbound_origin(r: SomeReference[_]):
    pass

# #33498: Variadics can't infer types for function pointers
fn indirect_function(x: Int):  pass
fn take_variadic_pack[*ArgTypes: AnyType](*args: *ArgTypes):  pass

# CHECK-LABEL: call_variadic_pack_with_function
fn call_variadic_pack_with_function():
  # CHECK: [[FP:%.*]] = kgen.create_closure[!lit.generator<("x": !Int) -> !kgen.none>: @parameters::@"indirect_function
  # CHECK: lit.call {{.*}}take_variadic_pack
  var x = take_variadic_pack(indirect_function)


# MOCO-1065: Crash handling self conditional conformance inference.
@value
struct MOCO1065[
    is_mutable: Bool, //,
    T: CollectionElement,
    o: Origin[is_mutable]._mlir_type,
]:
    fn __init__(out self: MOCO1065[UInt8, o], ref [o] string: Empty):
        pass

fn test_MOCO1065[p: Empty](t: Empty):
    var s = MOCO1065(t)
    alias a = MOCO1065(p)


### Complex dependent type inference problem.
@value
struct DepValue[a: Int]: pass
struct DepUser[b: Int]:
    fn foo(self):
        # This should infer
        var x : DepUser[2] = self.xyz(DepValue[1]())
    fn xyz(self, rhs: DepUser) -> __type_of(rhs): pass
    @implicit
    fn __init__[x: Int](value: DepValue[x], out result: DepUser[x+1]):
        pass


# Ensure we can infer a variadic parameter from inside an incoming
# parameter-value.

fn infer_variadic[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
    T: __type_of(Tuple[*ArgTypes]),
]():
    pass


# CHECK-LABEL:     lit.fn @"test_infer_variadic()"
fn test_infer_variadic():
    # CHECK: lit.call @parameters::@"infer_variadic{{.*}}"<:variadic<!AnyType> [
    # CHECK-SAME: [!Int,
    # CHECK-SAME: [!Bool,
    # CHECK-SAME: :meta<!lit.struct<#Tuple <:variadic<!AnyType>
    infer_variadic[Tuple[Int, Bool]]()
