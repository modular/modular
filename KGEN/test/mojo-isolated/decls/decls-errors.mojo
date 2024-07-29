# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Functions
##===----------------------------------------------------------------------===##

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
  testAsyncVoid() # expected-warning {{awaitable 'Coroutine[None, {}]' value was never awaited}}
  testAsyncInt() # expected-warning {{awaitable 'Coroutine[Int, {}]' value was never awaited}}


struct ThingWithStaticMethod:
   @staticmethod
   fn splat(x: Int): # expected-note {{function declared here}}
     pass

fn testThingWithStaticMethod():
  # expected-error @+1 {{invalid call to 'splat': argument #0 cannot be converted from 'FloatLiteral' to 'Int'}}
  ThingWithStaticMethod.splat(4.0)


def top_level_fn(a: Int):
    # expected-error @below {{TODO: closures cannot have parameters}}
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
    var decorator: Int

    @decorator # expected-error {{cannot use a dynamic value in decorator}}
    fn on_message():
        pass


@value
struct MemType:
    pass

# COM: Issue https://github.com/modularml/modular/issues/37758 where the
# COM: key test is that the below is not crashing due to assertion violation.
# expected-error @+1 {{expected ':' in function definition}}
fn missingColon(x: Int)
  return x


##===----------------------------------------------------------------------===##
# Named Results
##===----------------------------------------------------------------------===##


@__named_result(out)
# expected-error @below {{named results can only be used on functions with in-memory results, result type 'Int' is register-passable}}
fn regpassable_result() -> Int:
  pass


##===----------------------------------------------------------------------===##
# Default Arguments, VarArgs, and Packs
##===----------------------------------------------------------------------===##

# COM: Issue https://github.com/modularml/mojo/issues/1091
fn missing_arg_type_or_default(
    a: Int = 9,
    # expected-error @+2 {{required positional argument follows optional positional argument}}
    # expected-error @+1 {{'fn' argument type must be specified}}
    b,
    c: Int,  # expected-error {{required positional argument follows optional positional argument}}
    d: Int = 0,
    # expected-error @+2 {{required positional argument follows optional positional argument}}
    # expected-error @+1 {{'fn' argument type must be specified}}
    e,
    # expected-error @+1 {{'fn' argument type must be specified}}
    **kwargs,
):
    pass

def missing_default(
    a=9,
    b,  # expected-error {{equired positional argument follows optional positional argument}}
    c=0,
    d,  # expected-error {{required positional argument follows optional positional argument}}
):
    pass

# expected-error @+1 {{use of unknown declaration 'unknown'}}
fn defaultArgumentUnknownDeclaration(a: Int = unknown): pass

# expected-error @+1 {{cannot use a dynamic value in default argument}}
fn defaultArgumentReferencesArgument(a: Int = 0, b: Int = a): pass

# expected-error @+1 {{cannot implicitly convert 'FloatLiteral' value to 'Int'}}
fn defaultArgumentBadType(a: Int = 1.0): pass

# expected-error @+1 {{inout arguments may not have defaults}}
fn byref_default(inout x: Int = 2): pass

# expected-error @below {{'**' marker must be at end of argument list}}
fn starStarLast(**a: Int, b: Int): pass

# expected-error @below {{'**' marker must be at end of argument list}}
fn twoStarStar(**a: Int, **b: Int): pass

# expected-error @+1 {{expected argument name}}
fn starSpaceStar(* *a: Int): pass

# expected-error @+1 {{variadic arguments may not have defaults}}
fn noDefaultVariadics(*a: Int = 42): pass

# expected-note @+1 {{function declared here}}
fn exampleVariadic(a: FloatLiteral, *b: Int): pass
# expected-note @+1 {{function declared here}}
fn exampleByRefVariadic(a: FloatLiteral, inout *b: Int): pass
# expected-note @+1 {{function declared here}}
fn parameterizedVariadic[T: __mlir_type.`!kgen.type`](*args: T): pass

fn ownedPack[*Ts: AnyType](owned *args: *Ts): pass
fn ownedVariadic(owned *args: Inner): pass
fn ownedVariadicReg(owned *args: WrongType): pass


# expected-note @+1 {{struct declared here}}
struct ParameterizedStruct[T: __mlir_type.`!kgen.type`]:
    # expected-note @+1 {{function declared here}}
    def __init__(inout self, *args: T):
        pass

@value
struct TestTuple[*Ts: AnyTrivialRegType]:
    # expected-note @+1 {{function declared here}}
    fn test[i: Int, j: Int](self):
        pass

fn badCalls(arg: Int):
  # expected-error @+1 {{argument #1 cannot be converted from 'FloatLiteral' to 'Int'}}
  exampleVariadic(1.0, 1.0)
  # expected-error @+1 {{argument #3 cannot be converted from 'FloatLiteral' to 'Int'}}
  exampleVariadic(1.0, 1, 2, 1.0)

  var x: Int
  var y: FloatLiteral
  # expected-error @+1 {{invalid call to 'exampleByRefVariadic': argument #2 must be mutable in order to pass to a mutating argument}}
  exampleByRefVariadic(1.0, x, arg)
  # expected-error-re @+1 {{invalid call to 'exampleByRefVariadic': l-value of type 'FloatLiteral' cannot be converted to reference of type 'Int'}}
  exampleByRefVariadic(1.0, x, y)
  # expected-error @+1 {{argument #2 must be mutable in order to pass to a mutating argument}}
  exampleByRefVariadic(1.0, x, 1)

  # The user hasn't provided any arguments that could be used to infer `T`.
  # expected-error @below {{could not deduce parameter 'T' of callee 'parameterizedVariadic'}}
  # expected-note @below {{failed to infer parameter 'T', parameter isn't used in any argument}}
  parameterizedVariadic()
  # expected-error @below {{could not deduce parameter 'T' of parent struct 'ParameterizedStruct'}}
  # expected-note @below {{parameter isn't used in any argument}}
  var z = ParameterizedStruct()

  # We can't infer `T` with two arguments of different types.
  # expected-error @below {{invalid call to 'parameterizedVariadic': could not deduce parameter 'T' of callee 'parameterizedVariadic'}}
  # expected-note @below {{failed to infer parameter 'T', parameter inferred to two different values: 'Int' and 'FloatDyn'}}
  parameterizedVariadic(1, 2.0)

  # expected-error @below {{invalid call to 'test': could not deduce parameter 'j' of callee 'test'}}
  # expected-note @below {{failed to infer parameter 'j', parameter isn't used in any argument}}
  TestTuple[Int, FloatLiteral]().test[1]()

fn badError(a: ParameterizedStruct[Int]):
  # expected-error @+1 {{cannot implicitly convert 'ParameterizedStruct[Int]' value to 'ParameterizedStruct[Bool]'}}
  var b: ParameterizedStruct[Bool] = a

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

# expected-error @+1 {{unexpected token in expression}}
fn invalidStarExpression(*x: *): pass

# expected-error @+1 {{pack argument type list must reference a variadic list}}
fn invalidPackType(*x: *Int): pass

fn invalidParameterPack[*Ts: AnyType]():
  @parameter
  # expected-error @+2 {{expected a type, not a value}}
  # expected-error @+1 {{parameters may not be variadic packs}}
  fn invalid[*Us: *Ts](): pass

# expected-error @+2 {{only variadic arguments' types can be unpacked}}
# expected-note @+1 {{'x' is not a variadic argument}}
fn invalidArgumentUnpack[*Ts: AnyType](x: *Ts): pass

# expected-error @+1 {{argument already has a convention specified}}
fn invalidOwned(owned inout x: Int): pass

# expected-note @+1 {{function declared here}}
fn examplePack[*Ts: AnyType](*args: *Ts):
  pass

fn packArgOverload():
  pass

fn packArgOverload(x: Int):
  pass


# expected-note @+1 {{function declared here}}
fn first_and_rest[T: AnyTrivialRegType, *Ts: AnyType](*values: *Ts):
    pass

fn badPackCalls(value: Int):
  # expected-error @+1 {{invalid call to 'examplePack': callee with non-empty variadic pack argument expects 1 positional operand, but 2 were specified}}
  examplePack[Int](1, 2)
  # expected-error @+1 {{invalid call to 'examplePack': callee with non-empty variadic pack argument expects 2 positional operands, but 1 was specified}}
  examplePack[Int, FloatLiteral](1)
  # expected-error-re @+1 {{invalid call to 'examplePack': argument #1 cannot be converted from 'index' to 'FloatLiteral'}}
  examplePack[Int, FloatLiteral](1, Int(2).value)
  # expected-warning @below {{could not infer parameter type for this value, because it is not concrete}}
  # expected-error @below {{invalid call to 'examplePack': callee with non-empty variadic pack argument expects 0 positional operands, but 1 was specified}}
  examplePack(packArgOverload)
  # expected-error @below {{invalid call to 'first_and_rest': could not deduce parameter 'T' of callee 'first_and_rest'}}
  # expected-note @below {{failed to infer parameter 'T', parameter isn't used in any argument}}
  first_and_rest(value)

struct TestPackErrorMessage[*Ts: AnyType]:
    # expected-error @below {{'self' argument must have type 'TestPackErrorMessage[Ts]', but actually has type 'VariadicPack[0, args, AnyType, Ts]'}}
    fn __init__(*args: *Ts):
         pass

# expected-error @+1 {{variadic pack elements declared as 'AnyTrivialRegType' are removed, please declare elements as 'AnyType' instead of 'AnyTrivialRegType'}}
fn badAnyRegPack[*Ts: AnyTrivialRegType](*args: *Ts):
  pass

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
def fn_redecl2() -> FloatLiteral: pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: argument #0 cannot be converted from 'TestOverloading' to 'Int'}}
# expected-note @below {{candidate not viable: expected at most 1 positional argument, got 2}}
fn overloadIntFloat32(a: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note-re @below {{candidate not viable: argument #0 cannot be converted from 'TestOverloading' to 'FloatDyn'}}
# expected-note @below {{candidate not viable: expected at most 1 positional argument, got 2}}
fn overloadIntFloat32(a: FloatDyn): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: missing 1 required positional argument: 'b'}}
# expected-note-re @below {{candidate not viable: argument #1 cannot be converted from 'FloatDyn' to 'Int'}}
fn overloadIntFloat32(a: Int, b: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: missing 1 required positional argument: 'b'}}
# expected-note @below {{argument #1 must be mutable in order to pass to a mutating argument}}
fn overloadIntFloat32(a: Int, inout b: FloatDyn): pass

# expected-note @below {{candidate not viable: missing 2 required positional arguments: 'b', 'c'}}
# expected-note @below {{candidate not viable: missing 1 required positional argument: 'c'}}
# expected-note @below {{candidate declared here}}
fn overloadIntFloat32(a: Int, inout b: FloatDyn, c: Int, *args: Int): pass

struct TestOverloading:
  var a: Int   # expected-note {{cannot overload with this non-function definition}}
  fn a(self):  # expected-error {{invalid redefinition of 'a'}}
    pass

  fn test(self, a: Int, b: FloatDyn):
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
    # expected-error @below {{invalid call to 'bar': missing 1 required positional argument: 'f'}}
    a.bar()


# expected-note @+1 {{function declared here}}
fn takesAtLeastOneInt(x: Int, *y: Int): pass
fn badTakesAtLeastOneInt():
  # expected-error @+1 {{invalid call to 'takesAtLeastOneInt': missing 1 required positional argument: 'x'}}
  takesAtLeastOneInt()


# COM: Issue #23007
# expected-note @+1 {{function declared here}}
fn too_few_pos_only(a: Int, b: Int, /, msg: Int = 2): pass

fn test_too_few_pos_only(a: Int, msg: Int = 3):
  # expected-error @+1 {{invalid call to 'too_few_pos_only': missing 1 required positional argument: 'b'}}
  too_few_pos_only(a, msg=msg)


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

# COM: https://github.com/modularml/mojo/issues/1530
# COM: Do not crash when explicitly unbound parameter cannot be deduced due to missing arguments.
struct Parametric[a: Int]: pass

fn takes_same_arg_types[x: Int](a: Parametric[x], b: Parametric[x]): pass

fn test_param_deduction_failure[
    func: fn[y: Int] (c: Parametric[y], d: Parametric[y]) -> None,
](u: Int, v: Int):
    # expected-error @+1 {{cannot read from discard pattern '_'}}
    takes_same_arg_types[_](u)

    # expected-error @+1 {{cannot read from discard pattern '_'}}
    takes_same_arg_types[_](u, v)

    # expected-error @+1 {{missing 1 required positional argument: 'd'}}
    func[_](u)

    # TODO: This note is because we're not inferring signatures correctly
    # expected-error @below {{invalid indirect call: could not deduce parameter 'y' of callee 'callee'}}
    # expected-note @below {{failed to infer parameter 'y', parameter isn't used in any argument}}
    func[_](u, v)

struct InitOverloaded:
  # expected-note @below {{argument #1 cannot be converted from 'StringLiteral' to 'Int'}}
  # expected-note @below {{argument #1 cannot be converted from 'Parametric[1]' to 'Int'}}
  fn __init__(inout self, a: Int): pass
  # expected-note @below {{argument #1 cannot be converted from 'StringLiteral' to 'index'}}
  # expected-note @below {{argument #1 cannot be converted from 'Parametric[1]' to 'index'}}
  fn __init__(inout self, a: int): pass

fn testOverloadInitError(a: InitOverloaded, b: Parametric[1], c: Int):
  # expected-error @+1 {{cannot construct 'InitOverloaded' with itself, you can remove the constructor call}}
  _ = InitOverloaded(a)

  # expected-error @+1 {{no matching function in initialization}}
  _ = InitOverloaded(b)

  # This is ok
  _ = InitOverloaded(c)

  # expected-error @+1 {{no matching function in initialization}}
  _ = InitOverloaded("foo")



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

@staticmethod # expected-error {{only methods on structs may be declared static}}
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

@register_passable  # expected-error {{unsupported decorator on this statement}}
trait NoDecorators:
    pass

##===----------------------------------------------------------------------===##
# @deprecated
##===----------------------------------------------------------------------===##

@deprecated("use of deprecated struct 'DeprecatedStruct'")
# expected-note @below {{'DeprecatedStruct' declared here}}
struct DeprecatedStruct:
    pass

@deprecated("deprecated overload")
# expected-note @below {{'foobar' declared here}}
fn foobar():
    pass

# expected-warning @below {{use of deprecated struct 'DeprecatedStruct'}}
fn foobar(value: DeprecatedStruct):
    pass


fn deprecated_function():
   # expected-warning @below {{deprecated overload}}
   foobar()


from imported_module import DeprecatedInAnotherModule


# expected-warning @below {{use of deprecated struct 'DeprecatedInAnotherModule'}}
fn use_deprecated_import(value: DeprecatedInAnotherModule):
    pass


# expected-error @below {{@deprecated requires a warning message}}
@deprecated
fn no_message():
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

  # expected-error @+1 {{'__add__' requires 2 operands}}
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
  # expected-error @+1 {{'self' in struct '__init__' must be passed 'inout'}}
  def __init__(self): pass

  # expected-error @+1 {{'self' argument must have type 'WrongType', but actually has type 'Int'}}
  fn __init__(inout self: Int): pass

  # expected-error @+1 {{existing value argument must be passed as borrowed}}
  fn __copyinit__(inout self, inout existing: Self): pass

  # TODO: Should err.
  fn __copyinit__(inout self, existing: Int): pass

  # expected-error @+1 {{'@register_passable' types may not have a '__moveinit__' method, they are always movable by copying a register}}
  fn __moveinit__(inout self, owned existing: Self): pass


struct WrongSelfType[a: Int]:
  # expected-error @+1 {{'self' argument must have type 'WrongSelfType[a]', but actually has type 'Int'}}
  fn badMethod(self: Int): pass
  fn goodMethod(inout self: WrongSelfType[a]): pass

  # Issue #13358
  # expected-error @+1 {{'__copyinit__' requires 2 operands}}
  fn __copyinit__(inout self, other: Self, moar: Int): pass

  # expected-error @+1 {{'__add__' requires 2 operands}}
  fn __add__(self): pass

  fn __pow__(self, exp: Int): pass

  fn __pow__(self, exp: Int, mod: Int): pass

  # expected-error @+1 {{'__pow__' requires at least 2 operands}}
  fn __pow__(self): pass

  # expected-error @+1 {{'__pow__' requires at most 3 operands}}
  fn __pow__(self, exp: Int, mod: Int, extra: Int): pass

# Issue #6587: [Lit] Recursive constructors crash kgen
struct BadInit[size: __mlir_type.index]:
  fn __init__(inout self, elem: BadInit[Int(1).value]):
    var x : __mlir_type[`!pop.simd<`, size, `, FloatDyn>`]
    # expected-error @+1 {{cannot implicitly convert 'simd<size, FloatDyn>' value to 'BadInit[size]'}}
    self = x

struct StructWithField:
  var field: __mlir_type.index

# Issue #6879: Qualified lookup is looking up names wrong
fn unqualifiedNameLookup(a: StructWithField):
  # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
  a.badPropertyError

  # expected-error @+1 {{StructWithField' value has no attribute 'badPropertyError'}}
  StructWithField.badPropertyError

  # expected-error @+1 {{'EverythingIsWrongTrait' value has no attribute 'value'}}
  EverythingIsWrongTrait.value

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
  # expected-error @+1 {{trivial types may not have a '__copyinit__' method, they are always trivially copyable}}
  fn __copyinit__(inout self, existing: Self): pass
  # expected-error @+1 {{trivial types may not have a '__del__' method, they are always trivially destroyable}}
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
    var s1 = not_nested_struct()  # ok
    struct S2[*Ts: AnyType]: # expected-error {{struct inside a function not supported here}}
        fn __init__(inout self, *args: *Ts):
            pass
    var s2 = S2() # In issue https://github.com/modularml/modular/issues/12598 this was crashing.

# https://github.com/modularml/modular/issues/33557
struct HasBadCtor:
    var v: Int
    fn __init__(inout self, v: Int) -> Self: # expected-error {{'__init__' result type must be elided (or None)}}
        self.v = v
def useBadCtor():
    # Note that the key thing we're checking for here is that this does NOT have
    # a spurious error about HasBadCtor not being constructable from IntLiteral
    var fromBadCtor = HasBadCtor(123)

struct NotRegisterPassable:
    fn __init__(inout self):
        pass

# https://github.com/modularml/modular/issues/34551
# Don't crash on emitting methods when the struct itself is erroneous.
@value
@register_passable
struct Outer34551: # expected-error {{all members of '@register_passable' struct must themselves be '@register_passable'}}
    var _inner: NotRegisterPassable # expected-note {{'_inner' declared with type 'NotRegisterPassable'}}
    fn __init__(inout self):
        self._inner = NotRegisterPassable()
    # The key point of this test is that these errors break an invariant needed
    # for emission, so previously it would crash while emitting this __del__.
    fn __del__(owned self): # expected-error {{cannot transfer value into destination, because 'Outer34551' doesn't implement `__moveinit__`}}
        _ = self._inner ^

##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

trait EverythingIsWrongTrait:
    var value: Int # expected-error {{fields in traits are not supported yet}}

    fn trait_fn_has_body(self: Self): # expected-error {{unexpected function body in trait function declaration, use `...`}}
        var t = 1

    fn trait_fn_no_dot_dot_dot(self: Self): # expected-error {{expected body statements; use 'pass' if none is required}}

    trait NestedTrait: # expected-error {{nested trait not supported here}}
        ...

    # expected-note @below {{function declared here}}
    fn parametric[x: Int](self): ...

    struct NestedStruct: # expected-error {{nested struct in a trait not supported here}}
        pass

trait TraitWithParams[T: AnyTrivialRegType]: # expected-error {{TODO: trait declarations do not support parameters yet}}
    ...

fn bad_trait_params[T: EverythingIsWrongTrait](x: T):
  # expected-error @below {{invalid call to 'parametric': could not deduce parameter 'x' of callee 'parametric'}}
  # expected-note @below {{failed to infer parameter 'x', parameter isn't used in any argument}}
  x.parametric()

trait Shape(Copyable, Movable):
	fn area(self) -> int:
	    ...

@value
struct ShapeContainer:
    var shape: Shape # expected-error {{TODO: dynamic traits not supported yet, please use a compile time generic instead of 'Shape'}}

##===----------------------------------------------------------------------===##
# Struct/Trait conformance check failure
##===----------------------------------------------------------------------===##

trait CFMTrait: # expected-note {{trait 'CFMTrait' declared here}}
    # expected-note @below {{no 'f1' candidates have type 'fn(self: CFMStructFail) -> None'}}
    # expected-note @below {{required function 'f1' is not implemented}}
    fn f1(self: Self):
        pass

    @staticmethod
    fn f2(): # expected-note {{required function 'f2' is not implemented}}
        pass

# struct implements CFMTrait but does not have f2().
@register_passable("trivial")
struct CFMStructFail(CFMTrait): # expected-error {{struct 'CFMStructFail' does not implement all requirements for 'CFMTrait'}}
  fn f1(self, x: Int): # expected-note {{candidate declared here with type 'fn(self: CFMStructFail, x: Int) -> None'}}
    pass

@register_passable("trivial")
struct NoTraits: # expected-note {{'NoTraits' does not implement all requirements for 'CFMTrait'}}
    pass

fn trait_fn[T: CFMTrait]():
    pass

fn invalid_trait_bind():
    trait_fn[NoTraits]() # expected-error {{cannot bind type 'NoTraits' to trait 'CFMTrait'}}

fn non_copyable_trait[T: CFMTrait](value: T):
    var copy = value # expected-error {{'T' is not copyable because it has no '__copyinit__'}}


fn trait_fn_infer[T: CFMTrait](x: T): # expected-note {{function declared here}}
    pass

# expected-error @+1 {{no 'f1' candidates have type 'fn(self: CFMStructFail) -> None'}}
fn dont_crash_pvalue_convert(x: CFMStructFail):
    # expected-error @below {{invalid call to 'trait_fn_infer': could not deduce parameter 'T' of callee 'trait_fn_infer'}}
    # expected-note @below {{failed to infer parameter 'T', argument type 'CFMStructFail' does not conform to trait 'CFMTrait'}}
    trait_fn_infer(x)

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
  # expected-note @below {{no 'f' candidates have type 'fn[Int](self: UseTraitWithIntParamOnMethod) -> None'}}
  fn f[n: Int](self):
    ...
# expected-error @below {{struct 'UseTraitWithIntParamOnMethod' does not implement all requirements for 'TraitWithIntParamOnMethod'}}
struct UseTraitWithIntParamOnMethod(TraitWithIntParamOnMethod):
  # expected-note @below {{candidate declared here with type 'fn[Bool](self: UseTraitWithIntParamOnMethod) -> None'}}
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
from imported_module import DTypePointer # expected-note {{previous definition here}}
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


# Issue https://github.com/modularml/mojo/issues/1675
# Ensure @value fails gracefully in the presence of duplicate field names.
@value
struct BadStruct:
    var b: int  # expected-note {{previous definition here}}
    var b: int  # expected-error {{invalid redefinition of 'b'}}


# Also ensure that @value doesn't fail if a method/alias shadows it.
@value
struct OtherBadStruct:
    # expected-note @below {{previous definition here}}
    # expected-note @below {{cannot overload with this non-function definition}}
    var b: int
    alias b = `0`  # expected-error {{invalid redefinition of 'b'}}

    fn b(inout self):  # expected-error {{invalid redefinition of 'b'}}
        pass


fn test_bad_struct():
    _ = BadStruct(`1`)
    _ = OtherBadStruct(`2`)

##===----------------------------------------------------------------------===##
# Top Level Code
##===----------------------------------------------------------------------===##

fn top_level_func() raises -> Int:
   pass

fn use_error(e: Error):
   pass

# expected-error @below {{cannot call function that may raise in a context that cannot raise}}
# expected-note @below {{try surrounding the call in a 'try' block}}
var np = top_level_func()

# expected-error @below {{'try' must be contained in a function}}
try:
    var np2 = top_level_func()
except e:
    # expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
    use_error(e)


fn top_level_func_param[p: Int]():
    pass

alias a = 100
# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
top_level_func_param[a]()

var y = 7
# expected-error @below {{TODO: expressions are not yet supported at the file scope level}}
y += 1
