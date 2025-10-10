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

@invalidDec # expected-error {{use of unknown declaration 'invalidDec'}}
def bad_decorator(): pass

fn decorator_on_var():
    @invalidDec
    var DecoratedVar: Int # expected-error {{'var' statement does not allow decorators}}

# expected-error @+1 {{decorators must be on their own line, not ahead of a statement}}
@always_inline def same_line_decorator(): pass

# @parameter if causes confusing indentation error message
# https://github.com/modularml/modular/issues/19163
fn someFn():
    # expected-error @below {{decorators must be on their own line, not ahead of a statement}}
    @decorator if True:
        pass

fn someFn2():
        # expected-error @below {{orphaned decorator not associated with a declaration or statement}}
        @decorator
    if True: # expected-error {{unknown tokens at the end of a declaration}}
        pass


# ===----------------------------------------------------------------------=== #
# @staticmethod
# ===----------------------------------------------------------------------=== #

@staticmethod # expected-error {{only methods on structs may be declared static}}
def not_a_struct_method(): pass


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
    @implicit # expected-error {{'@implicit' may only be applied to '__init__' methods}}
    fn foo(mut self): pass
    @implicit # expected-error {{'@implicit' requires an argument to convert from}}
    fn __init__(out self): pass
    @implicit # expected-error {{'@implicit' initializers must accept a single argument value}}
    fn __init__(out self, x: Int, y: Int): pass
    @implicit # expected-error {{'@implicit' may only be applied to '__init__' methods}}
    fn __copyinit__(out self, other: Self): pass


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

# expected-error @+2 {{unexpected function body in extern function declaration, use `...`}}
@extern("add_one")
fn my_extern_add_one(x: Int) -> Int:
    return x + 1

struct HasExtern:
  # expected-error @+1 {{@extern cannot be applied to a method}}
  @extern("add_one_struct")
  fn my_extern_struct_add_one(self, x: Int) -> Int:
    ...


# ===----------------------------------------------------------------------=== #
# @__llvm_arg_metadata
# ===----------------------------------------------------------------------=== #

# expected-error @below {{LLVM arg metadata requires an argument name}}
@__llvm_arg_metadata()
fn llvm_arg_meta_no_arg[x: Int](a: Int, b: Int):
    pass

# expected-error @below {{First argument of LLVM arg metadata must be an argument name}}
@__llvm_arg_metadata(1 + 1)
fn llvm_arg_meta_wrong_type[x: Int](a: Int, b: Int):
    pass

# expected-error @below {{No argument named c}}
@__llvm_arg_metadata(c, myMeta)
fn llvm_arg_meta_wrong_name[x: Int](a: Int, b: Int):
    pass


# ===----------------------------------------------------------------------=== #
# Struct decorators
# ===----------------------------------------------------------------------=== #

@invalidDec  # expected-error {{use of unknown declaration 'invalidDec'}}
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
