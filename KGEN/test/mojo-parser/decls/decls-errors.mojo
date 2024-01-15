# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# Functions
##===----------------------------------------------------------------------===##

def func():
  never_declared_fn() # expected-error {{use of unknown declaration 'never_declared_fn'}}

# expected-error @+1 {{special function '__add__' must have 2 operands}}
fn __add__(): pass
# expected-error @+1 {{special function must be a method}}
fn __sub__(self: Int, a: Int): pass

fn mutArgAndImplicit(a: Int):
  a = a  # expected-error {{expression must be mutable in assignment}}
  c = a  # expected-error {{use of unknown declaration 'c', 'fn' declarations require explicit variable declarations}}

fn missingColon()  # expected-error {{expected ':' in function definition}}
  # Don't get confused by comments or blank lines!

  var x = 1 # expected-error {{could not find builtin 'IntLiteral' type}}

# Missing colon after fn definition complains about function effects
# https://github.com/modularml/modular/issues/23359
def missingColon2() # expected-error {{missing ':' at end of function signature}}
  func()

# expected-error @below {{expected parameter name}}
# expected-error @below {{unexpected token in expression}}
fn missingArgumentName(*: Int): pass

# expected-error @below {{expected parameter name}}
# expected-error @below {{unexpected token in expression}}
fn missingParameterName[: Int](): pass

# expected-error @+1 {{use of unknown declaration 'InvalidType'}}
fn badPropertyError(a: InvalidType):
  _ = a.value   # Should not produce a follow-on error.
  return

struct NotBoolConvertible: pass
# expected-note @+1 {{function declared here}}
fn test_bool_context(a: NotBoolConvertible): pass

fn voidReturningFn(): pass
fn badCall():
  # expected-error @+1 {{invalid call to 'test_bool_context': argument #0 cannot be converted from 'None' to 'NotBoolConvertible'}}
  test_bool_context(voidReturningFn())


fn missing_ret_val() -> __mlir_type.index:
  return # expected-error {{cannot implicitly convert 'None' value to 'index' in return value}}

fn ret_type_mismatch() -> __mlir_type.index:
  return 4.0 # expected-error {{cannot implicitly convert 'FloatLiteral' value to 'index' in return value}}

async fn testAsyncVoid(): pass
async fn testAsyncInt() -> Int: return 42

fn callsWith():
  testAsyncVoid() # expected-warning {{awaitable 'Coroutine[None]' value was never awaited}}
  testAsyncInt() # expected-warning {{awaitable 'Coroutine[Int]' value was never awaited}}


struct ThingWithStaticMethod:
   @staticmethod
   fn splat(x: Int): # expected-note {{function declared here}}
     pass

fn testThingWithStaticMethod():
  # expected-error @+1 {{invalid call to 'splat': argument #0 cannot be converted from 'FloatLiteral' to 'Int'}}
  ThingWithStaticMethod.splat(4.0)


def top_level_fn(a: Int):
    # expected-error @below {{nonparametric capturing closure cannot have input parameters}}
    fn bar[b: Int]() -> Int:
      return a


def use_non_copyable_type(a: ThingWithStaticMethod):
  pass

def test_use_non_copyable_type(owned b: ThingWithStaticMethod):
  use_non_copyable_type(b^)

# Issue #14191
# expected-error @+1 {{unexpected tokens after decorator, each need to be on their own line}}
@always_inline wqeqwe
fn issue14191() -> Int:
    return 1

fn issue1242():
    try:
        fn decorator(function: fn(abc:Int) capturing -> None) escaping:
           print("calling a func")

        @decorator # expected-error {{cannot use a dynamic value in decorator}}
        fn on_message(abc:Int) -> None:
            print(abc)
    except e:
        print(e)


@value
struct MemType:
    pass

# FIXME(#26008): Async functions with memory-only do not work.
# expected-error @below {{TODO: async functions do not support memory-only results yet}}
async fn async_mem_result() -> MemType:
  pass

##===----------------------------------------------------------------------===##
# Default Arguments, VarArgs, and Packs
##===----------------------------------------------------------------------===##

# COM: Issue https://github.com/modularml/mojo/issues/1091
fn missing_arg_type_or_default(
    a: Int = 9,
    # expected-error @+2 {{non-default argument follows default argument}}
    # expected-error @+1 {{'fn' argument type must be specified}}
    b,
    c: Int,  # expected-error {{non-default argument follows default argument}}
    d: Int = 0,
    # expected-error @+2 {{non-default argument follows default argument}}
    # expected-error @+1 {{'fn' argument type must be specified}}
    e,
):
    pass

def missing_default(
    a=9,
    b,  # expected-error {{non-default argument follows default argument}}
    c=0,
    d,  # expected-error {{non-default argument follows default argument}}
):
    pass

# expected-error @+1 {{use of unknown declaration 'unknown'}}
fn defaultArgumentUnknownDeclaration(a: Int = unknown): pass

# expected-error @+1 {{cannot use a dynamic value in default argument}}
fn defaultArgumentReferencesArgument(a: Int = 0, b: Int = a): pass

# expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int' in default argument}}
fn defaultArgumentBadType(a: Int = 1.0): pass

# expected-error @+1 {{inout arguments may not have defaults}}
fn byref_default(inout x: Int = 2): pass

# expected-error @+1 {{'**' marker must be at end of argument list}}
fn starStarLast(**a: Int, b: Int): pass

# expected-error @+1 {{expected parameter name}}
fn starSpaceStar(* *a: Int): pass

# expected-error @+1 {{variadic arguments may not have defaults}}
fn noDefaultVariadics(*a: Int = 42): pass

# expected-note @+1 {{function declared here}}
fn exampleVariadic(a: Float32, *b: Int): pass
# expected-note @+1 {{function declared here}}
fn exampleByRefVariadic(a: Float32, inout *b: Int): pass
# expected-note @+1 {{function declared here}}
fn parameterizedVariadic[T: __mlir_type.`!kgen.anyregtype`](*args: T): pass

# expected-error @+1 {{'owned' arguments cannot be variadic}}
fn ownedPack[*Ts: __mlir_type.`!kgen.anyregtype`](owned *args: *Ts): pass
# expected-error @+1 {{'owned' arguments cannot be variadic}}
fn ownedVariadic(owned *args: Inner): pass
# expected-error @+1 {{'owned' arguments cannot be variadic}}
fn ownedVariadicReg(owned *args: WrongType): pass


# expected-note @+1 {{struct declared here}}
struct ParameterizedStruct[T: __mlir_type.`!kgen.anyregtype`]:
    # expected-note @+1 {{function declared here}}
    def __init__(inout self, *args: T):
        pass

@value
struct TestTuple[*Ts: AnyRegType]:
    # expected-note @+1 {{function declared here}}
    fn test[i: Int, j: Int](self):
        pass

fn badCalls(arg: Int):
  # expected-error @+1 {{argument #1 cannot be converted from 'FloatLiteral' to 'Int'}}
  exampleVariadic(1.0, 1.0)
  # expected-error @+1 {{argument #3 cannot be converted from 'FloatLiteral' to 'Int'}}
  exampleVariadic(1.0, 1, 2, 1.0)

  var x: Int
  var y: Float32
  # expected-error @+1 {{invalid call to 'exampleByRefVariadic': argument #2 must be mutable in order to pass as a by-ref argument}}
  exampleByRefVariadic(1.0, x, arg)
  # expected-error-re @+1 {{invalid call to 'exampleByRefVariadic': l-value of type 'SIMD[f32, 1]' cannot be converted to reference of type 'Int'}}
  exampleByRefVariadic(1.0, x, y)
  # expected-error @+1 {{argument #2 must be mutable in order to pass as a by-ref argument}}
  exampleByRefVariadic(1.0, x, 1)

  # FIXME(#11803): These diagnostics could be improved.
  # The user hasn't provided any arguments that could be used to infer `T`.
  # expected-error @+1 {{callee expects 1 input parameter, but 0 were specified}}
  parameterizedVariadic()
  # expected-error @+1 {{could not deduce parameter #0 ('T') of parent struct ParameterizedStruct}}
  let z = ParameterizedStruct()
  # We can't infer `T` with two arguments of different types.
  # expected-error @+1 {{callee expects 1 input parameter, but 0 were specified}}
  parameterizedVariadic(1, 2.0)

  # expected-error @below {{callee expects 3 input parameters, but 2 were specified}}
  TestTuple[Int, Float32]().test[1]()

fn badError(a: ParameterizedStruct[Int]):
  # expected-error @+1 {{cannot implicitly convert 'ParameterizedStruct[Int]' value to 'ParameterizedStruct[Bool]' in 'let' initializer}}
  let b: ParameterizedStruct[Bool] = a

# expected-note @below {{candidate declared here}}
fn overloadedFunc(x: Int): pass
# expected-note @below {{candidate declared here}}
fn overloadedFunc(x: Int, y: Int): pass

# expected-note @below {{function declared here}}
fn takeFuncArgument(f: Int): pass

fn callWithOverloadedArg():
  # expected-error @below {{invalid call to 'takeFuncArgument': argument #0 cannot be converted from unknown overload to}}
  # expected-error @below {{cannot convert function to non-function type 'Int'}}
  # expected-note @below {{try resolving the overloaded function first}}
  takeFuncArgument(overloadedFunc)

# expected-note @below {{function declared here}}
fn takeGenericResultFn[T: AnyRegType](f: fn() -> T): pass

fn returnMemType() -> MemType:
    return MemType()

fn passMemTypeResultGeneric():
    # expected-error @below {{invalid call to 'takeGenericResultFn': argument #0 cannot be converted from 'fn() -> MemType'}}
    # expected-note @below {{memory-only type bound to generic result type: payload returns 'MemType' by reference}}
    takeGenericResultFn[MemType](returnMemType)

# expected-error @+1 {{unexpected token in expression}}
fn invalidStarExpression(*x: *): pass

# expected-error @+1 {{only variadic types may be unpacked}}
fn invalidPackType(*x: *Int): pass

fn invalidParameterPack[*Ts: __mlir_type.`!kgen.anyregtype`]():
  @parameter
  # expected-error @+1 {{parameters may not be variadic packs}}
  fn invalid[*Us: *Ts](): pass

# expected-error @+2 {{only variadic arguments' types can be unpacked}}
# expected-note @+1 {{'x' is not a variadic argument}}
fn invalidArgumentUnpack[*Ts: __mlir_type.`!kgen.anyregtype`](x: *Ts): pass

# expected-error @+1 {{argument already has a convention specified}}
fn invalidOwned(owned inout x: Int): pass

# expected-note @+1 {{function declared here}}
fn examplePack[*Ts: __mlir_type.`!kgen.anyregtype`](*args: *Ts):
  pass

fn packArgOverload():
  pass

fn packArgOverload(x: Int):
  pass

fn directly_pass_pack(pack: __mlir_type.`!kgen.pack<[index]>`):
  pass

# expected-note @+1 {{function declared here}}
fn first_and_rest[T: AnyRegType, *Ts: AnyRegType](*values: *Ts):
    pass

fn badPackCalls(value: Int):
  # expected-error @+1 {{invalid call to 'examplePack': callee expects 1 argument, but 2 were specified}}
  examplePack[Int](1, 2)
  # expected-error @+1 {{invalid call to 'examplePack': callee expects 2 arguments, but 1 was specified}}
  examplePack[Int, Float32](1)
  # expected-error-re @+1 {{invalid call to 'examplePack': argument #1 cannot be converted from 'index' to 'SIMD[{{.*}}f32{{.*}}]'}}
  examplePack[Int, Float32](1, Int(2).value)
  # expected-warning @below {{could not infer parameter type for this value, because it is not concrete}}
  # expected-error @below {{invalid call to 'examplePack': callee expects 0 arguments, but 1 was specified}}
  examplePack(packArgOverload)
  # expected-error @below {{invalid call to 'first_and_rest': callee expects 2 input parameters, but 0 were specified}}
  first_and_rest(value)

##===----------------------------------------------------------------------===##
# Keyword Arguments
##===----------------------------------------------------------------------===##

# expected-error @+1 {{keyword-only arguments not supported yet}}
def kw1(a, *, *, b): pass # expected-error {{cannot have two '*' markers in the same argument list}}
def kw2(a, /, /, b): pass # expected-error {{cannot have two '/' markers in the same argument list}}
# expected-error @+1 {{keyword-only arguments not supported yet}}
def kw3(a, /, *, b): pass # OK
# expected-error @+1 {{keyword-only arguments not supported yet}}
def kw4(a, *, /, b): pass # expected-error {{cannot specify '/' marker after '*' marker}}
def kw5(/, a):       pass # expected-error {{'/' marker cannot be used at the start of the argument list}}
def kw6(a, *):       pass # expected-error {{'*' marker is not allowed at end of argument list}}
# expected-error @+1 {{keyword-only arguments not supported yet}}
def kw7(*a: Int, *b: Int): pass # expected-error {{cannot have two '*' markers in the same argument list}}
# expected-error @+1 {{keyword-only arguments not supported yet}}
def kw8[*Ts: __mlir_type.`!kgen.anyregtype`](*a: *Ts, *b: *Ts): pass # expected-error {{cannot have two '*' markers in the same argument list}}
fn kw9(*a: Int, b: Int): pass # expected-error {{keyword-only arguments not supported yet}}

##===----------------------------------------------------------------------===##
# Function Overloading
##===----------------------------------------------------------------------===##

# expected-note @+1 {{previous definition here}}
def fn_redecl(): pass
# expected-error @+1 {{redefinition of function 'fn_redecl' with identical signature}}
def fn_redecl(): pass

# expected-note @+1 {{previous definition here}}
def fn_redecl2() -> Int: pass
# expected-error @+1 {{redefinition of function 'fn_redecl2' cannot overload on return type only}}
def fn_redecl2() -> Float32: pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #0 cannot be converted from 'TestOverloading' to 'Int'}}
# expected-note @below {{candidate not viable: callee expects 1 argument}}
fn overloadIntFloat32(a: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note-re @below {{candidate not viable: argument #0 cannot be converted from 'TestOverloading' to 'SIMD[{{.*}}f32{{.*}}]'}}
# expected-note @below {{candidate not viable: callee expects 1 argument}}
fn overloadIntFloat32(a: Float32): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: callee expects 2 arguments}}
# expected-note-re @below {{candidate not viable: argument #1 cannot be converted from 'SIMD[{{.*}}f32{{.*}}]' to 'Int'}}
fn overloadIntFloat32(a: Int, b: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: callee expects 2 arguments}}
# expected-note @below {{argument #1 must be mutable in order to pass as a by-ref argument}}
fn overloadIntFloat32(a: Int, inout b: Float32): pass

# expected-note @below {{callee expects at least 3 arguments, but 1 was specified}}
# expected-note @below {{callee expects at least 3 arguments, but 2 were specified}}
# expected-note @below {{candidate declared here}}
fn overloadIntFloat32(a: Int, inout b: Float32, c: Int, *args: Int): pass

struct TestOverloading:
  var a: Int   # expected-note {{cannot overload with this non-function definition}}
  fn a(self):  # expected-error {{invalid redefinition of 'a'}}
    pass

  fn test(self, a: Int, b: Float32):
    # expected-error @+1 {{cannot form a reference to overloaded declaration}}
    var bad = overloadIntFloat32

    # expected-error @+1 {{no matching function in call}}
    overloadIntFloat32(self)
    # expected-error @+1 {{no matching function in call}}
    overloadIntFloat32(a, b)


# Test that static methods don't get dispatched if their first arg is self type.
struct StructWithStaticMethod:
    fn __init__(inout self): pass

    # expected-note @+2 {{function declared here}}
    @staticmethod
    fn bar(inout f: StructWithStaticMethod): pass


fn test_static_overload():
    var a = StructWithStaticMethod()
    # expected-error @below {{call to 'bar': callee expects 1 argument, but 0 were specified}}
    a.bar()


# expected-note @+1 {{function declared here}}
fn takesAtLeastOneInt(x: Int, *y: Int): pass
fn badTakesAtLeastOneInt():
  # expected-error @+1 {{callee expects at least 1 argument, but 0 were specified}}
  takesAtLeastOneInt()


# COM: Issue #23007
# expected-note @+1 {{function declared here}}
fn too_few_pos_only(a: Int, b: Int, /, msg: Int = 2): pass

fn test_too_few_pos_only(a: Int, msg: Int = 3):
  # expected-error @+1 {{callee expects at least 2 positional arguments, but 1 was specified}}
  too_few_pos_only(a, msg=msg)


alias int = __mlir_type.index

alias `1` = __mlir_attr.`1 : index`
alias `2` = __mlir_attr.`2 : index`

# COM: Issue #23007
# expected-note @+1 {{function declared here}}
fn missing_args(a: int, b: int, c: int = `2`, d: int = `2`): pass

fn test_missing_args():
  # expected-error @+1 {{invalid call to 'missing_args': missing 2 required positional arguments: 'a', 'b'}}
  _ = missing_args(c=`1`, d=`1`)


struct ConvertibleFromInt:
  fn __init__(inout self, value: Int):
    pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #1 cannot be converted from 'ConvertibleFromInt' to 'Int'}}
fn ambiguousConversions(a: ConvertibleFromInt, b: Int): pass
# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #0 cannot be converted from 'ConvertibleFromInt' to 'Int'}}
fn ambiguousConversions(a: Int, b: ConvertibleFromInt): pass

fn testAmbiguousConversions(a: Int, b: ConvertibleFromInt):
  ambiguousConversions(a, b) # ok
  ambiguousConversions(b, a) # ok
  # expected-error @+1 {{ambiguous call to 'ambiguousConversions', each candidate requires 1 implicit conversion, disambiguate with an explicit cast}}
  ambiguousConversions(a, a)
  # expected-error @+1 {{no matching function in call}}
  ambiguousConversions(b, b)

  var localFn = testAmbiguousConversions  # ok
  localFn(a, b)
  localFn(a, a)
  localFn(1, b)
  # expected-error @+1 {{invalid indirect call: argument #0 cannot be converted from 'ConvertibleFromInt' to 'Int'}}
  localFn(b, b)

##===----------------------------------------------------------------------===##
# Decorators
##===----------------------------------------------------------------------===##

@decorator  # expected-error {{use of unknown declaration 'decorator'}}
struct DecoratedStruct: pass

fn decoratorTest():
  @decorator
  var DecoratedVar: Int # expected-error {{'var' statement does not allow decorators}}

@invalidDec # expected-error {{use of unknown declaration 'invalidDec'}}
def BadDecorator(): pass

@staticmethod # expected-error @+1 {{only methods on structs may be declared static}}
def StaticMethod(): pass

struct DecoratorSameLine:
  # expected-error @below {{decorators must be on their own line, not ahead of a statement}}
  @staticmethod def StaticMethod(): pass

# @parameter if causes confusing indentation error message
# https://github.com/modularml/modular/issues/19163
fn someFn():
    # expected-error @below {{decorators must be on their own line, not ahead of a statement}}
    @parameter if True:
        pass

fn someFn2():
    # expected-error @below {{orphaned decorator not associated with a declaration or statement}}
    @parameter
  if True: # expected-error {{unknown tokens at the end of a declaration}}
    pass

##===----------------------------------------------------------------------===##
# Structs
##===----------------------------------------------------------------------===##

# expected-error @below {{recursive reference to declaration}}
# expected-note @below {{previously used here}}
struct Rec[param: Rec]:
  pass

# expected-error @+1 {{'def' statement must be on its own line}}
struct Struct: def foo(inout self): pass

struct ReturnFromStruct:
  # expected-error @+1 {{cannot return from this context}}
  return 42

struct ReDef: pass # expected-note {{previous definition here}}
struct ReDef: pass # expected-error {{invalid redefinition of 'ReDef'}}

struct StructMemberRedefinition:
  var x : __mlir_type.index  # expected-note {{previous definition here}}
  var x : __mlir_type.index  # expected-error {{invalid redefinition of 'x'}}

struct SpecialFunctions:
  # expected-error @+1 {{'__new__' is not supported on structs}}
  fn __new__() -> Self:
    pass

  # expected-error @+1 {{special function '__add__' must have 2 operands}}
  fn __add__(self):
    pass

  # This is ok, SpecialFunctions may be a reference semantic struct like Object.
  # Issue #7573: Cannot have LValue or RValue overloads of special methods
  fn __iadd__(a: SpecialFunctions, b: SpecialFunctions): pass

  # expected-error @+1 {{'__iadd__' result type must be elided (or None)}}
  fn __iadd__(inout self, rhs: SpecialFunctions) -> SpecialFunctions: pass

  fn failures(self):
    self+self # Supports this, even though it isn't valid.  Shouldn't crash.
    self*self # expected-error {{'SpecialFunctions' does not implement the '__mul__' method}}

  # expected-error @+1 {{'__del__' cannot be declared as raising an exception}}
  fn __del__(owned self) raises:
     pass

@register_passable
struct WrongType:
  # expected-error @+1 {{'__init__' result type must be 'WrongType'}}
  def __init__(self): pass

  # expected-error @+1 {{'__init__' result type must be 'WrongType'}}
  def __init__() -> Int: pass

  # expected-error @+1 {{special function '__copyinit__' must have 1 operand}}
  fn __copyinit__(inout self, inout existing: Int): pass

  # expected-error @+1 {{self argument cannot be passed by reference}}
  fn __copyinit__(inout self) -> WrongType: pass

  # expected-error @+1 {{'__moveinit__' is not supported for @register_passable types, they are always movable by copying a register}}
  fn __moveinit__(owned self) -> Self: pass


struct WrongSelfType[a: Int]:
  # expected-error @+1 {{'self' argument must have type 'WrongSelfType[a]' but actually has type 'Int'}}
  fn badMethod(self: Int): pass
  fn goodMethod(inout self: WrongSelfType[a]): pass

  # Issue #13358
  # expected-error @+1 {{special function '__copyinit__' must have 2 operands}}
  fn __copyinit__(inout self, other: Self, moar: Int): pass

  # expected-error @+1 {{special function '__add__' must have 2 operands}}
  fn __add__(self): pass

  fn __pow__(self, exp: Int): pass

  fn __pow__(self, exp: Int, mod: Int): pass

  # expected-error @+1 {{special function '__pow__' must have at least 2 operands}}
  fn __pow__(self): pass

  # expected-error @+1 {{special function '__pow__' must have at most 3 operands}}
  fn __pow__(self, exp: Int, mod: Int, extra: Int): pass

# Issue #6587: [Lit] Recursive constructors crash kgen
struct BadInit[size: __mlir_type.index]:
  fn __init__(inout self, elem: BadInit[Int(1).value]):
    var x : __mlir_type[`!pop.simd<`, size, `, Float32>`]
    # expected-error @+1 {{cannot implicitly convert 'simd<size, Float32>' value to 'BadInit[size]' in assignment}}
    self = x

  # expected-error @+1 {{'__init__' result type must be elided (or None)}}
  fn __init__(inout self) -> Self: pass

struct StructWithField:
  var field: __mlir_type.index

# Issue #6879: Qualified lookup is looking up names wrong
fn unqualifiedNameLookup(a: StructWithField):
  # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
  a.badPropertyError

  # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
  StructWithField.badPropertyError

  # expected-error @+1 {{cannot access instance field 'field' without an instance of 'StructWithField'}}
  StructWithField.field

struct DirectInstanceReference:
  var value: Int
  fn fxn(self):
    # expected-error @+1 {{cannot access instance field 'value' directly; did you mean 'self.'?}}
    var xx = value

  @staticmethod
  fn stat():
    _ = fxn  # expected-error {{cannot access method 'fxn' directly; did you mean 'Self.'?}}

  fn direct_ref(self):
    fxn(self) # expected-error {{cannot access method 'fxn' directly; did you mean 'self.'?}}
    stat() # expected-error {{cannot access method 'stat' directly; did you mean 'Self.'?}}


fn field_indexes(a: DirectInstanceReference):
  a.badField = 42 # expected-error {{'DirectInstanceReference' value has no attribute 'badField'}}

struct MLIRAttrWithinStruct:
  # expected-error @below {{MLIR attribute is not a TypedAttr}}
  __mlir_attr.`#index<cmp_predicate eq>`


# In register structs may only have stored properties of other in-reg values.
struct InMemStruct: pass

# expected-error @+2 {{all members of '@register_passable' struct must themselves be '@register_passable'}}
@register_passable
struct InRegStruct:
  var x: Int # ok
  var y: InMemStruct # expected-note {{'y' declared with type 'InMemStruct'}}

struct OtherInMemStruct:
  var x: Int # ok
  var y: InMemStruct # ok


@register_passable("trivial")
struct InvalidMember:
  var x: __mlir_type.index
  # expected-error @+1 {{'@register_passable("trivial")' types may not have a '__copyinit__' method}}
  fn __copyinit__(self) -> Self: pass
  # expected-error @+1 {{'@register_passable("trivial")' types may not have a '__del__' method}}
  fn __del__(owned self): pass

def noop():  # expected-error {{expected body statements; use 'pass' if none is required}}

struct BadDtor1:
  fn __del__(self): # expected-error {{self argument must be 'owned'}}
    pass

struct BadDtor:
  fn __init__(inout self): pass
  fn __del__[x: Int](owned self):
    pass

fn bad_destructors():
  var x = BadDtor()


@value # expected-error {{'@value' cannot synthesize members: 'x' has non-copyable, non-movable type 'InMemStruct'}}
struct CantSynthesize:
  var x : InMemStruct # expected-note {{'x' declared here}}


@value # expected-error {{'@value' cannot synthesize members of struct 'ResolveErrorIsBubbled'}}
struct ResolveErrorIsBubbled:
   var x: Int
   fn __init__(inout self, x: unknown): # expected-error {{use of unknown declaration 'unknown'}}
      pass

fn function_with_struct():
  struct Foo: # expected-error {{struct inside a function not supported here}}
    var x: Int

# https://github.com/modularml/modular/issues/12598
struct not_nested_struct[*Ts: AnyType]:
    fn __init__(inout self, *args: *Ts):
        pass
fn function_with_struct2():
    let s1 = not_nested_struct()  # ok
    struct S2[*Ts: AnyType]: # expected-error {{struct inside a function not supported here}}
        fn __init__(inout self, *args: *Ts):
            pass
    let s2 = S2() # In issue https://github.com/modularml/modular/issues/12598 this was crashing.

struct TypeGetItem:
    fn __getitem__(inout self, key: Int): # expected-note {{function declared here}}
        pass

fn bad_metatype_access():
    var val = TypeGetItem
    val[1] # expected-error {{invalid call to '__getitem__': l-value of type 'TypeGetItem' cannot be converted to reference of type 'TypeGetItem}}


struct BadRefItem:
    fn __init__(inout self): pass
    fn __refitem__(inout self, key: Int) -> Int:
        return key

fn access_BadRefItem():
    var val = BadRefItem()
    _ = val[1] # expected-error {{the '__refitem__' method on 'BadRefItem' returned a value of 'Int', expected a reference}}


##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

trait EverythingIsWrongTrait:
    var value: Int # expected-error {{fields in traits are not supported yet}}

    fn trait_fn_has_body(self: Self): # expected-error {{unexpected function body in trait function declaration, use `...`}}
        let t = 1

    fn trait_fn_no_dot_dot_dot(self: Self): # expected-error {{expected body statements; use 'pass' if none is required}}

    trait NestedTrait: # expected-error {{nested trait not supported here}}
        ...

    # expected-note @below {{function declared here}}
    fn parametric[x: Int](self): ...

    struct NestedStruct: # expected-error {{nested struct in a trait not supported here}}
        pass

trait TraitWithParams[T: AnyRegType]: # expected-error {{TODO: trait declarations do not support parameters yet}}
    ...

fn bad_trait_params[T: EverythingIsWrongTrait](x: T):
  x.parametric() # expected-error {{invalid call to 'parametric': callee expects 1 input parameter, but 0 were specified}}

##===----------------------------------------------------------------------===##
# Struct/Trait conformance check failure
##===----------------------------------------------------------------------===##

trait CFMTrait: # expected-note {{trait 'CFMTrait' declared here}}
    fn f1(self: Self): # expected-note {{no 'f1' candidates have type 'fn(self = CFMStructFail) -> None'}}
        pass

    @staticmethod
    fn f2(): # expected-note {{required function 'f2' is not implemented}}
        pass

# struct implements CFMTrait but does not have f2().
@register_passable("trivial")
struct CFMStructFail(CFMTrait): # expected-error {{struct 'CFMStructFail' does not implement all requirements for 'CFMTrait'}}
  fn f1(self, x: Int): # expected-note {{candidate declared here with type 'fn(self = CFMStructFail, x = Int) -> None'}}
    pass

@register_passable("trivial")
struct NoTraits: # expected-note {{'NoTraits' does not implement 'CFMTrait'}}
    pass

fn trait_fn[T: CFMTrait]():
    pass

fn invalid_trait_bind():
    trait_fn[NoTraits]() # expected-error {{cannot bind type 'NoTraits' to trait 'CFMTrait'}}

fn non_copyable_trait[T: CFMTrait](value: T):
    let copy = value # expected-error {{'T' is not copyable because it has no '__copyinit__'}}


fn trait_fn_infer[T: CFMTrait](x: T): # expected-note {{function declared here}}
    pass

fn dont_crash_pvalue_convert(x: CFMStructFail):
    trait_fn_infer(x) # expected-error {{invalid call to 'trait_fn_infer': callee expects 1 input parameter, but 0 were specified}}

trait GrandFather: # expected-note {{trait 'GrandFather' declared here}}
    fn foo(self): # expected-note {{required function 'foo' is not implemented}}
        ...

trait Father(GrandFather): # expected-note {{inherited through 'Father' here}}
    pass

# expected-error @below {{struct 'MissingInheritedFn' does not implement all requirements for 'GrandFather'}}
# expected-warning @below {{'MissingInheritedFn' already inherits from 'GrandFather'}}
# expected-note @below {{inherited through 'Father' here}}
struct MissingInheritedFn(Father, GrandFather):
    pass

# expected-warning @below {{'InheritsTwice' already inherits from 'Father'}}
# expected-note @below {{previously inherited here}}
struct InheritsTwice(Father, Father):
    fn foo(self):
        pass


# https://github.com/modularml/mojo/issues/1399
# Parser crash when trait implementation parameters don't match the definition
# expected-note @below {{trait 'TraitWithIntParamOnMethod' declared here}}
trait TraitWithIntParamOnMethod:
  # expected-note @below {{no 'f' candidates have type 'fn[Int](self = UseTraitWithIntParamOnMethod) -> None'}}
  fn f[n: Int](self):
    ...
# expected-error @below {{caller input parameter #0 has type }}
# expected-error @below {{struct 'UseTraitWithIntParamOnMethod' does not implement all requirements for 'TraitWithIntParamOnMethod'}}
struct UseTraitWithIntParamOnMethod(TraitWithIntParamOnMethod):
  # expected-note @below {{candidate declared here with type 'fn[Bool](self = UseTraitWithIntParamOnMethod) -> None'}}
  fn f[n: Bool](self):
    pass

##===----------------------------------------------------------------------===##
# Class
##===----------------------------------------------------------------------===##

class SomeClass:  # expected-error {{classes are not supported yet}}
  pass

##===----------------------------------------------------------------------===##
# Exported Functions
##===----------------------------------------------------------------------===##

@export
def valid_name():
  ...

# expected-error @+1 {{@export requires a string specifying the name of the exported symbol}}
@export(1)
def export_me():
  ...

# expected-note @+1 {{previous export here}}
@export("my_foo")
def foo():
  ...

# expected-error @+1 {{invalid re-export of my_foo}}
@export("my_foo")
def bar():
  ...

# expected-error @+1 {{my+foo is not a valid C identifier}}
@export("my+foo", ABI="C")
def bad_name():
  ...

# expected-note @+1 {{previous export here}}
@export
def func_overloaded(x: Int):
  ...

# expected-error @+1 {{invalid re-export of func_overloaded}}
@export
def func_overloaded(x: Bool):
  ...


# Issue #12090
from memory.unsafe import DTypePointer # expected-note {{previous definition here}}
struct DTypePointer: # expected-error {{invalid redefinition of 'DTypePointer'}}
    pass

# Issue #13321.
struct copy_init_def:
  var field: Int

  # expected-error @+1 {{cannot define '__copyinit__' as 'def'; 'def' implicitly raises}}
  def __copyinit__(inout self, existing: Self):
    self.field = existing.field

struct copy_init_raises:
  # expected-error @+1 {{'__copyinit__' cannot be declared as raising an exception}}
  fn __copyinit__(inout self, existing: Self) raises:
     pass


# Order of declaration processing.
# https://github.com/modularml/mojo/issues/235
@value
struct Inner:
    pass

@value
@register_passable
struct Outer: # expected-error {{all members of '@register_passable' struct must themselves be '@register_passable'}}
    var inner: Inner # expected-note {{'inner' declared with type 'Inner'}}


@value # expected-error {{cannot synthesize members: 'value' has non-copyable, non-movable type 'T'}}
struct AnyTypeMember[T: AnyType]:
    var value: T # expected-note {{'value' declared here}}

##===----------------------------------------------------------------------===##
# Top Level Code
##===----------------------------------------------------------------------===##

fn top_level_func() raises:
   pass

# expected-error @below {{cannot call function that may raise in a context that cannot raise}}
# expected-note @below {{try surrounding the call in a 'try' block}}
let np = top_level_func()

# expected-error @below {{'try' must be contained in a function}}
try:
    let np2 = top_level_func()
except e:
    # expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
    _ = e

alias a = 100
# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
constrained[a == 10]()

var y = 7
# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
y += 1
