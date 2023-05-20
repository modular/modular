# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s -I %S/../mojo-examples/

from prolog import DType, Error, F32, Int, object

##===----------------------------------------------------------------------===##
# Lexical Issues
##===----------------------------------------------------------------------===##

# https://github.com/modularml/modular/issues/4181
struct Issue4181IndentWeirdness[dt: DType]:
  var b : Int
    # expected-error @+1 {{definition isn't on its own line at the correct indentation}}
    def f():
      pass

# Failed to parse due to indentation.
fn issue_6291(
    val: __mlir_type.index
) -> __mlir_type.index:
    return val

fn testIndentation6291[index: __mlir_type.index](
    ptr: __mlir_type.`!pop.pointer<!pop.scalar<index>>`):
  var result = __mlir_op.`pop.load`[
            alignment : __mlir_attr.`1: index`,
            _type : __mlir_type.`!pop.scalar<index>`
](ptr)

# Failed to parse doc strings.
struct struct_issue_6526:
    """
    foo
    """
    pass

fn fn_issue_6526():
    """
    foo
    """
    pass

# This file contains parsing related bugs.

fn bracketError1():
  _ = ] # expected-error {{unexpected token in expression}}

fn bracketError2():
  _ = [[1, 2], }# expected-error {{unexpected token in expression}}

##===----------------------------------------------------------------------===##
# Return
##===----------------------------------------------------------------------===##

def foo():
# expected-error @+1 {{unexpected token in expression}}
  return pass

return 32 # expected-error {{cannot return from this context}}

##===----------------------------------------------------------------------===##
# If / While
##===----------------------------------------------------------------------===##

def elif_parse_error(a: Bool):
  if a:
    pass
 elif a: # expected-error {{unexpected token in expression}}
    pass
  else:
    pass

struct NotBoolConvertible:
  fn __copyinit__(inout self, existing: Self):
    pass

def test_bool_context(a: NotBoolConvertible):
  if a: # expected-error {{NotBoolConvertible' does not implement the '__bool__' method}}
     pass

fn test_if_decorator(a: Bool):
  @not_good() # expected-error {{unsupported decorator on 'if' statement}}
  if a:
    pass

  @parameter
  if 1 != 0:
    pass
  elif a:  # expected-error {{cannot use a dynamic value in '@parameter if' condition}}
    pass

##===----------------------------------------------------------------------===##
# For
##===----------------------------------------------------------------------===##

struct my_iter_no_len:
    fn __init__(inout self): pass
    fn __next__(inout self) -> Int: return 0


struct MyList_range_no_len:
    fn __init__(inout self): pass
    fn __iter__(self) -> my_iter_no_len: return my_iter_no_len()


struct my_iter_no_next:
    fn __init__(inout self): pass
    fn __len__(self) -> Int: return 0


struct MyList_range_no_next:
    fn __init__(inout self): pass
    fn __iter__(self) -> my_iter_no_next: return my_iter_no_next()


struct MyList_no_iter:
    fn __init__(inout self): pass


struct my_iter_wrong_int:
    fn __init__(inout self): pass
    fn __next__(inout self) -> Int: return 0
    fn __len__(self: my_iter_wrong_int) -> F32: return 0.0


struct MyList_invalid_boxed_type:
    fn __init__(inout self): pass
    fn __iter__(self) -> my_iter_wrong_int: return my_iter_wrong_int()


fn main():
    let my_list_no_len = MyList_range_no_len()
    let my_list_no_next = MyList_range_no_next()
    let my_list_no_iter = MyList_no_iter()
    let my_list_invalid_int = MyList_invalid_boxed_type()

    # expected-error @+1 {{'my_iter_no_len' does not implement the '__len__' method}}
    for item in my_list_no_len:
        pass

    # expected-error @+1 {{'my_iter_no_next' does not implement the '__next__' method}}
    for item in my_list_no_next:
        pass

    # expected-error @+1 {{'MyList_no_iter' does not implement the '__iter__' method}}
    for item in my_list_no_iter:
        pass

    # expected-error @+1 {{'SIMD[f32, 1]' does not implement the '__as_mlir_index' method}}
    for item in my_list_invalid_int:
        pass

    # expected-error @+1 {{expected 'in' after target identifier. Note that target lists are not yet supported.}}
    for key, item in my_list_no_next:
        pass

##===----------------------------------------------------------------------===##
# Raise
##===----------------------------------------------------------------------===##

def raisingFunction():
    pass

fn callRaisingFunction():
    raisingFunction() # expected-error {{cannot call function that may raise in a context that cannot raise}}

fn cannotReRaise():
    raise # expected-error {{no contextual exception to reraise}}

fn cannotRaise(err: Error):
    raise err # expected-error {{cannot raise error in a context that cannot raise}}

# Issue #12358
fn raise_bad_type() raises:
    raise 42  # expected-error {{cannot implicitly convert 'Int' value to 'Error' in raised value}}

# Issue #6909
# expected-error @below {{expected name for 'alias' declaration}}
# expected-note @below {{escape keyword 'True' with backticks to use it as an identifier}}
alias True = 42
