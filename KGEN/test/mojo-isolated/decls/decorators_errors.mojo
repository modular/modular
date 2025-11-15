# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

# ===----------------------------------------------------------------------=== #
# Function decorators
# ===----------------------------------------------------------------------=== #

# Issue #14191
# expected-error @+1 {{unexpected tokens after decorator, each need to be on their own line}}
@always_inline wqeqwe
fn issue14191() -> Int:
    return 1

fn issue1242():
    var decorator: Int

    @decorator # expected-error {{cannot use a dynamic value in decorator}}
    fn on_message(): pass

@invalid_dec # expected-error {{use of unknown declaration 'invalid_dec'}}
def unknown_decorator(): pass

fn decorator_on_statements():
    @invalid_dec
    var decorated_var: Int  # expected-error {{'var' statement in function body does not allow decorators}}

    @invalid_dec
    alias decorated_alias = 42  # expected-error {{'alias' statement in function body does not allow decorators}}

    @invalid_dec
    while True:  # expected-error {{'while' statement does not allow decorators}}
        pass

    @invalid_dec
    _ = 1 + 1  # expected-error {{statement does not allow decorators}}


# expected-error @+1 {{decorators must be on their own line, not ahead of a statement}}
@always_inline def same_line_decorator(): pass

# @parameter if causes confusing indentation error message
# https://github.com/modularml/modular/issues/19163
fn some_fn():
    # expected-error @below {{decorators must be on their own line, not ahead of a statement}}
    @decorator if True:
        pass

fn some_fn_2():
        # expected-error @below {{orphaned decorator not associated with a declaration or statement}}
        @decorator
    if True: # expected-error {{unknown tokens at the end of a declaration}}
        pass

@decorator[]  # expected-error {{invalid expression in decorator}}
fn bad_decorator_expression_1():
    pass

@decorator[]()  # expected-error {{invalid expression in decorator}}
fn bad_decorator_expression_2():
    pass

@567  # expected-error {{invalid expression in decorator}}
fn bad_decorator_expression_3():
    pass

# ===----------------------------------------------------------------------=== #
# @always_inline
# ===----------------------------------------------------------------------=== #

@always_inline("builtin", "nodebug")  # expected-error {{'@always_inline' may not have more than 1 operand, got 2}}
fn bad_always_inline_1():
    pass

@always_inline(123)  # expected-error {{'@always_inline' operand must be "nodebug" or "builtin"}}
fn bad_always_inline_2():
    pass

@always_inline("no_debug")  # expected-error {{'@always_inline' operand must be "nodebug" or "builtin"}}
fn bad_always_inline_3():
    pass

# ===----------------------------------------------------------------------=== #
# @staticmethod
# ===----------------------------------------------------------------------=== #

@staticmethod  # expected-error {{only methods on structs may be declared static}}
def not_a_struct_method(): pass

struct HasBadStaticMethod:
    @staticmethod()  # expected-error {{'@staticmethod' cannot have arguments}}
    fn bad_static_method_1(): pass

    @staticmethod("abc")  # expected-error {{'@staticmethod' cannot have arguments}}
    fn bad_static_method_2(): pass


# ===----------------------------------------------------------------------=== #
# @no_inline
# ===----------------------------------------------------------------------=== #

@no_inline()  # expected-error {{'@no_inline' cannot have arguments}}
fn bad_no_inline_1():
    pass

@no_inline("abc")  # expected-error {{'@no_inline' cannot have arguments}}
fn bad_no_inline_2():
    pass


# ===----------------------------------------------------------------------=== #
# @deprecated
# ===----------------------------------------------------------------------=== #

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


# ===----------------------------------------------------------------------=== #
# @implicit
# ===----------------------------------------------------------------------=== #

struct CheckImplicit:
    # expected-error @+1 {{'@implicit' may only be applied to '__init__' methods}}
    @implicit
    fn foo(mut self): pass

    # expected-error @+2 {{'@implicit' initializers must accept a single positional argument value}}
    @implicit
    fn __init__(out self): pass

    # expected-error @+2 {{'@implicit' initializers must accept a single positional argument value}}
    @implicit
    fn __init__(out self, x: Int, y: Int): pass

    # expected-error @+2 {{'@implicit' initializers must accept a single positional argument value}}
    @implicit
    fn __init__(out self, *, z: Int): pass

    # expected-error @+1 {{'@implicit' may only be applied to '__init__' methods}}
    @implicit
    fn __copyinit__(out self, other: Self): pass

    # expected-error @+1 {{'@implicit' may not have more than 1 operand, got 2}}
    @implicit(123, "abc")
    fn __init__(out self, a: Int): pass

    # expected-error @+1 {{'@implicit' may only have a keyword argument 'deprecated' with literal boolean value}}
    @implicit(123)
    fn __init__(out self, a: Int): pass

    # expected-error @+1 {{'@implicit' may only have a keyword argument 'deprecated' with literal boolean value}}
    @implicit(deprecated=123)
    fn __init__(out self, b: String): pass

    # expected-error @+1 {{'@implicit' may only have a keyword argument 'deprecated' with literal boolean value}}
    @implicit(foo=True)
    fn __init__(out self, c: Bool): pass

struct DeprecatedImplicitConversion:
    # expected-note @+2 {{implicit constructor for 'DeprecatedImplicitConversion' declared here}}
    @implicit(deprecated=True)
    fn __init__(out self, value: Int):
        pass

fn foo(y: DeprecatedImplicitConversion): pass

fn foo(z: String): pass

fn deprecated_implicit_conversion():
    # expected-warning @+1 {{deprecated implicit conversion from 'IntLiteral[1]' to 'DeprecatedImplicitConversion'}}
    _: DeprecatedImplicitConversion = 1

    # expected-warning @+1 {{deprecated implicit conversion from 'Int' to 'DeprecatedImplicitConversion'}}
    foo(Int(1))

# ===----------------------------------------------------------------------=== #
# @export
# ===----------------------------------------------------------------------=== #

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


# ===----------------------------------------------------------------------=== #
# @extern
# ===----------------------------------------------------------------------=== #

@extern  # expected-error {{'@extern' requires 1 argument}}
fn bad_extern_1(): ...

@extern()  # expected-error {{'@extern' requires 1 argument}}
fn bad_extern_2(): ...

@extern(123)  # expected-error {{'@extern' requires a string literal argument}}
fn bad_extern_3(): ...

@extern("bad_extern", "bad_extern_3")  # expected-error {{'@extern' requires 1 argument}}
fn bad_extern_4(): ...

# expected-error @+2 {{unexpected function body in extern function declaration, use `...`}}
@extern("add_one")
fn my_extern_add_one(x: Int) -> Int:
    return x + 1

struct HasExtern:
  # expected-error @+1 {{'@extern' cannot be applied to a method}}
  @extern("add_one_struct")
  fn my_extern_struct_add_one(self, x: Int) -> Int:
    ...

# ===----------------------------------------------------------------------=== #
# @__llvm_metadata
# ===----------------------------------------------------------------------=== #

@__llvm_metadata  # expected-error {{'@__llvm_metadata' requires operands}}
fn llvm_meta_no_arg_1[x: Int](a: Int, b: Int):
    pass

@__llvm_metadata()  # expected-error {{'@__llvm_metadata' requires operands}}
fn llvm_meta_no_arg_2[x: Int](a: Int, b: Int):
    pass


# ===----------------------------------------------------------------------=== #
# @__llvm_arg_metadata
# ===----------------------------------------------------------------------=== #

@__llvm_arg_metadata  # expected-error {{'@__llvm_arg_metadata' requires operands}}
fn llvm_arg_meta_no_arg_1[x: Int](a: Int, b: Int):
    pass

@__llvm_arg_metadata()  # expected-error {{'@__llvm_arg_metadata' requires operands}}
fn llvm_arg_meta_no_arg_2[x: Int](a: Int, b: Int):
    pass

# expected-error @+1 {{First argument of '@__llvm_arg_metadata' must be an argument name}}
@__llvm_arg_metadata(1 + 1)
fn llvm_arg_meta_wrong_type[x: Int](a: Int, b: Int):
    pass

# expected-error @+1 {{Function decorated by '@__llvm_arg_metadata' has no argument named 'c'}}
@__llvm_arg_metadata(c, myMeta)
fn llvm_arg_meta_wrong_name[x: Int](a: Int, b: Int):
    pass

# ===----------------------------------------------------------------------=== #
# Closure decorators
# ===----------------------------------------------------------------------=== #

fn outer_function():
    @__copy_capture  # expected-error {{'@__copy_capture' must have arguments}}
    @parameter()  # expected-error {{'@parameter' cannot have arguments}}
    fn copy_capture_no_args_1():
        pass

    @__copy_capture()  # expected-error {{'@__copy_capture' must have arguments}}
    @parameter("abc")  # expected-error {{'@parameter' cannot have arguments}}
    fn copy_capture_no_args_2():
        pass

    @__move_capture  # expected-error {{'@__move_capture' must have arguments}}
    @parameter
    fn move_capture_no_args_1():
        pass

    @__move_capture()  # expected-error {{'@__move_capture' must have arguments}}
    @parameter
    fn move_capture_no_args_2():
        pass

# ===----------------------------------------------------------------------=== #
# Struct decorators
# ===----------------------------------------------------------------------=== #

@invalid_dec  # expected-error {{use of unknown declaration 'invalid_dec'}}
struct BadStructDecorator: pass


struct DecoratorSameLine:
  # expected-error @+1 {{decorators must be on their own line, not ahead of a statement}}
  @staticmethod def same_line_decorator(): pass


@value # expected-error {{'@value' has been removed, please use '@fieldwise_init' and explicit `Copyable` and `Movable` conformances instead}}
struct LegacyValueDecorator:
  pass


@fieldwise_init # expected-error {{'FieldwiseInitExample' has an explicitly declared fieldwise initializer}}
struct FieldwiseInitExample[T: Movable]:
  var x: Int
  var y: T

  # expected-note @below {{initializer declared here}}
  fn __init__(out self, x: Int, y: T):
    pass


# ===----------------------------------------------------------------------=== #
# Trait decorators
# ===----------------------------------------------------------------------=== #

@decorator  # expected-error {{unrecognized body decorators}}
trait NoDecorators:
    pass
