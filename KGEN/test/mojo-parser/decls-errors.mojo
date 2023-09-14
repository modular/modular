# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics -split-input-file %s

from memory.unsafe import Pointer


##===----------------------------------------------------------------------===##
# Closures
##===----------------------------------------------------------------------===##

fn bind_fat_to_thin_target[g: fn(Int) -> Int](x: Int):
    pass


fn bind_fat_to_thin_main():
    let x = 4

    @parameter
    fn g(y: Int) -> Int:
        return x

    # expected-error @below {{cannot pass 'fn(y = Int) capturing -> Int' value, parameter expected 'fn(Int) -> Int'}}
    alias Bound = bind_fat_to_thin_target[g]
    Bound(3)

##===----------------------------------------------------------------------===##
# Var / Let
##===----------------------------------------------------------------------===##

def var_decl_without_type():
  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'ReturnFromStruct' in 'var' initializer}}
  var y : ReturnFromStruct = 1.0

  # expected-error @+1 {{declaration must have either a type or an initializer}}
  var x

  # expected-error @below {{cannot implicitly convert 'SIMD[f32, 16]' value to 'SIMD[f32, 8]' in 'let' initializer}}
  let z: SIMD[DType.float32, 8] = SIMD[DType.float32, 16]()

def var_decl():
  x = 123        # expected-note {{previous definition here}}
  var x : Int  # expected-error {{invalid redefinition of 'x'}}
  x+4   # no follow-on error.

def err():
  var localVar = 42
  var y : localVar  # expected-error {{cannot use a dynamic value in type specification}}

def missing_type_on_var_decl():
  var abc :
  pass # expected-error {{unexpected token in expression}}
  abc+abc

def bad_stmt_list():
  # expected-error @+1 {{'if' statement must be on its own line}}
  abc = 0; if abc != 0: pass

def cantFwdDeclarePlusEqual():
  # expected-error @+1 {{use of unknown declaration 'x'}}
  x += 4

fn badTypeErrorMessage():
  var x: Int
  # expected-error @+1 {{cannot use a dynamic value in type specification}}
  let ptr: Pointer[Int].address_of(x)

struct StructWithLets:
  let struct_thing : Int # expected-error {{'let' fields in structs are not supported yet}}


fn use_before_def():
    # expected-error @below {{use of unknown declaration 'x', 'fn' declarations require explicit variable declarations}}
    let y = x
    let x = 10

# Issue #18150: https://github.com/modularml/modular/issues/18150
fn self_reference():
    # expected-error @+1 {{use of unknown declaration 'num', 'fn' declarations require explicit variable declarations}}
    let num: Int = num + 2

##===----------------------------------------------------------------------===##
# Functions
##===----------------------------------------------------------------------===##

def func():
  never_declared_fn() # expected-error {{use of unknown declaration 'never_declared_fn'}}

# expected-error @+1 {{special function '__add__' must have 2 operands}}
fn __add__(): pass
# expected-error @+1 {{special function must be a method}}
fn __sub__(self: Int, a: Int): pass

# Test differences between fn and def.
fn noArgType(a: Int, b): pass # expected-error {{'fn' argument type must be specified}}

fn mutArgAndImplicit(a: Int):
  a = a  # expected-error {{expression must be mutable in assignment}}
  c = a  # expected-error {{use of unknown declaration 'c', 'fn' declarations require explicit variable declarations}}

fn missingColon()  # expected-error {{expected ':' in function definition}}
  # Don't get confused by comments or blank lines!

  var x = 1 # expected-error {{could not find builtin 'Int' type}}

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


# expected-error @+1 {{cannot return and raise the same type from a function}}
fn cant_raise_return(a: Error) raises -> Error:
  return a


def top_level_fn(a: Int):
    # expected-error @+2 {{nonparametric capturing closure cannot be marked @adaptive}}
    @adaptive
    fn adaptive_capturing_closure() -> Int:
        return a

    # expected-error @below {{nonparametric capturing closure cannot have input or result parameters}}
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

##===----------------------------------------------------------------------===##
# Default Arguments, VarArgs, and Packs
##===----------------------------------------------------------------------===##

# expected-error @+1 {{non-default argument follows default argument}}
fn nonDefaultArgumentFollowsDefaultArgument(a: Int = 0, b: Int): pass

# expected-error @+1 {{use of unknown declaration 'unknown'}}
fn defaultArgumentUnknownDeclaration(a: Int = unknown): pass

# expected-error @+1 {{use of unknown declaration 'a'}}
fn defaultArgumentReferencesArgument(a: Int = 0, b: Int = a): pass

# expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int' in default argument}}
fn defaultArgumentBadType(a: Int = 1.0): pass

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
fn parameterizedVariadic[T: __mlir_type.`!kgen.mlirtype`](*args: T): pass

struct ParameterizedStruct[T: __mlir_type.`!kgen.mlirtype`]:
    # expected-note @+1 {{function declared here}}
    def __init__(inout self, *args: T):
        pass

@value
struct TestTuple[*Ts: AnyType]:
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
  # expected-error-re @+1 {{l-value of type 'SIMD[{{.*}}f32{{.*}}]' cannot be converted to reference of type 'Int'}}
  exampleByRefVariadic(1.0, x, y)
  # expected-error @+1 {{argument #2 must be mutable in order to pass as a by-ref argument}}
  exampleByRefVariadic(1.0, x, 1)

  # FIXME(#11803): These diagnostics could be improved.
  # The user hasn't provided any arguments that could be used to infer `T`.
  # expected-error @+1 {{callee expects 1 input parameter but 0 were provided}}
  parameterizedVariadic()
  # expected-error @+1 {{callee expects 1 input parameter but 0 were provided}}
  let z = ParameterizedStruct()
  # We can't infer `T` with two arguments of different types.
  # expected-error @+1 {{callee expects 1 input parameter but 0 were provided}}
  parameterizedVariadic(1, 2.0)

  # expected-error @below {{callee expects 3 input parameters but 2 were provided}}
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
fn takeGenericResultFn[T: AnyType](f: fn() -> T): pass

@value
struct MemType:
    pass

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

fn invalidParameterPack[*Ts: __mlir_type.`!kgen.mlirtype`]():
  @parameter
  # expected-error @+1 {{parameters may not be variadic packs}}
  fn invalid[*Us: *Ts](): pass

# expected-error @+2 {{only variadic arguments' types can be unpacked}}
# expected-note @+1 {{'x' is not a variadic argument}}
fn invalidArgumentUnpack[*Ts: __mlir_type.`!kgen.mlirtype`](x: *Ts): pass

# expected-error @+1 {{argument already has a convention specified}}
fn invalidOwned(owned inout x: Int): pass

# expected-note @+1 {{function declared here}}
fn examplePack[*Ts: __mlir_type.`!kgen.mlirtype`](*args: *Ts):
  pass

fn packArgOverload():
  pass

fn packArgOverload(x: Int):
  pass

fn badPackCalls():
  # expected-error @+1 {{invalid call to 'examplePack': callee expects 1 argument, but 2 were specified}}
  examplePack[Int](1, 2)
  # expected-error @+1 {{invalid call to 'examplePack': callee expects 2 arguments, but 1 was specified}}
  examplePack[Int, Float32](1)
  # expected-error-re @+1 {{invalid call to 'examplePack': argument #1 cannot be converted from 'index' to 'SIMD[{{.*}}f32{{.*}}]'}}
  examplePack[Int, Float32](1, (2).value)
  # expected-warning @below {{could not infer parameter type for this value, because it is not concrete}}
  # expected-error @below {{invalid call to 'examplePack': callee expects 1 input parameter but 0 were provided}}
  examplePack(packArgOverload)

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
def kw8[*Ts: __mlir_type.`!kgen.mlirtype`](*a: *Ts, *b: *Ts): pass # expected-error {{cannot have two '*' markers in the same argument list}}
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


# expected-note @+1 {{function declared here}}
fn takesAtLeastOneInt(x: Int, *y: Int): pass
fn badTakesAtLeastOneInt():
  # expected-error @+1 {{callee expects at least 1 argument, but 0 were specified}}
  takesAtLeastOneInt()


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
  if True:
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

  # expected-error @+1 {{'__takeinit__' is not supported for @register_passable types, they are always movable by copying a register}}
  fn __takeinit__(inout self) -> Self: pass


struct WrongSelfType[a: Int]:
  # expected-error @+1 {{'self' argument must have type 'WrongSelfType[a]'}}
  fn badMethod(self: Int): pass
  fn goodMethod(inout self: WrongSelfType[a]): pass

  # Issue #13358
  # expected-error @+1 {{special function '__copyinit__' must have 2 operands}}
  fn __copyinit__(inout self, other: Self, moar: Int): pass

  # expected-error @+1 {{special function '__add__' must have 2 operands}}
  fn __add__(self): pass

# Issue #6587: [Lit] Recursive constructors crash kgen
struct BadInit[size: __mlir_type.index]:
  fn __init__(inout self, elem: BadInit[(1).value]):
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

##===----------------------------------------------------------------------===##
# 'main' Function
##===----------------------------------------------------------------------===##

# // -----

# expected-error @below {{expected 'main' function to have no arguments}}
fn main(arg: Int):
  return

# // -----

# expected-error @below {{expected 'main' function to have no arguments}}
def main(arg: Int):
  return

# // -----

# expected-error @below {{expected 'main' function returning object to be raising}}
fn main() -> object:
  return

# // -----

# expected-error @below {{expected 'main' function to return 'None'}}
fn main() -> Int:
  return 10

# // -----

# expected-error @below {{expected 'main' function to have no parameters}}
fn main[input: Int]():
  return

# // -----

# expected-error @below {{'main' can only be exported as 'main'}}
@export("foo")
fn main():
  return

# // -----

# expected-error @below {{only 'main' can be exported as 'main'}}
@export("main")
fn foo():
  return

# // -----

##===----------------------------------------------------------------------===##
# Top Level Code
##===----------------------------------------------------------------------===##

fn foo() raises:
   pass

# expected-error @below {{cannot call function that may raise in a context that cannot raise}}
# expected-note @below {{try surrounding the call in a 'try' block}}
let np = foo()

# expected-error @below {{'try' must be contained in a function}}
try:
    let np2 = foo()
except e:
    # expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
    print(e.value)

alias a = 100
# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
constrained[a == 10]()

var y = 7
# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
y += 1
