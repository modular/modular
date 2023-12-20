# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo -verify-diagnostics %s

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
    ptr: __mlir_type.`!kgen.pointer<!pop.scalar<index>>`):
  var result = __mlir_op.`pop.load`[
            alignment=__mlir_attr.`1: index`,
            _type=__mlir_type.`!pop.scalar<index>`
](ptr)

# This file contains parsing related bugs.

fn bracketError1():
  _ = ] # expected-error {{unexpected token in expression}}

fn bracketError2():
  _ = [[1, 2], }# expected-error {{unexpected token in expression}}


# Indentation errors
fn nothing(): pass

fn test_indentation1():
  nothing()   # expected-note {{indentation should match previous statement}}
    nothing() # expected-error {{statement has excess indentation}}

fn test_indentation2():
  nothing()
  if True:   # expected-note {{indentation should match previous statement}}
      nothing()
   nothing() # expected-error {{statement has excess indentation}}


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
 elif a: # expected-error {{unknown tokens at the end of a declaration}}
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
    fn __len__(self: my_iter_wrong_int) -> Float32: return 0.0


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

    # expected-error @+1 {{'SIMD[f32, 1]' does not implement the '__index__' method}}
    for item in my_list_invalid_int:
        pass

    # expected-error @+1 {{expected 'in' after target identifier. Note that target lists are not yet supported.}}
    for key, item in my_list_no_next:
        pass

# Issue #18599
fn spurious_for_loop_variable_unknown_decl():
  # expected-error @below {{'FloatLiteral' does not implement the '__iter__' method}}
  for i in 1.0:
    # Note that the bug in issue #18599 is that after the above error, another error
    # will be spuriously raised about i not being bound.  So the real check in
    # this test is that no further error is raised.
    _ = i

##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##

struct ExampleCM:
  fn __moveinit__(inout self, owned other: Self): pass
  fn __enter__(self) -> Int:
    return 42
  fn __exit__(self):
    pass # normal
  fn __exit__(self, err: Error) -> Bool:
    return True # Raise

def withUsingImmutableVariable(owned a: ExampleCM):
  # expected-note @below {{'x' declared here}}
  let x = 77
  # expected-error @below {{'x' is not a valid mutable variable for `with ... as` to target}}
  with a^ as x:
    pass

# External Issue #529 https://github.com/modularml/mojo/issues/529
def withWithNoColon(a: __mlir_type.index):
  # expected-error @below {{expected ':' after 'with' expression}}
  with a as b

# Issue #20143 https://github.com/modularml/modular/issues/20143
struct HasBadContextManagerExit:
  var x: Int
  fn __init__(inout self, x:Int):
    self.x = x
  fn __copyinit__(inout self, other:Self):
    self.x = other.x
  fn __enter__(self) -> Self:
      return self
  # Note that the __exit__ method takes 2 arguments, IE
  # `fn __exit__(self, err: Error) -> Bool:`
  # So this will have a failing __exit__ call when used in `with`.
  # expected-note @below {{function declared here}}
  fn __exit__(self) -> Bool:
      return True
def useBadContextManagerExit():
  # expected-error @below {{invalid call to '__exit__'}}
  with HasBadContextManagerExit(5) as bad:
      _ = bad.x

# Poor error when with context managers that take ownership in enter
# https://github.com/modularml/modular/issues/23100
struct BadCM: # expected-note {{'BadCM' declared here}}
  fn __init__(inout self): pass

  fn __enter__(owned self) -> Int:
    return 42
  fn __exit__(self):
    pass # normal
  fn __exit__(self, err: Error) -> Bool:
    return True # Raise

fn noop(a: Int): pass

fn testBadCM():
  # expected-error @+1 {{context manager of type 'BadCM' defines a consuming __enter__ method as well as an __exit__ method; either remove 'owned' from its '__enter__' method or remove the '__exit__' method}}
  with BadCM():
    pass



##===----------------------------------------------------------------------===##
# Raise
##===----------------------------------------------------------------------===##

def raisingFunction():
    pass

# expected-note @below {{or mark surrounding function as 'raises'}}
fn callRaisingFunction():
    # expected-error @below {{cannot call function that may raise in a context that cannot raise}}
    # expected-note @below {{try surrounding the call in a 'try' block}}
    raisingFunction()

fn cannotReRaise():
    # expected-error @below {{no contextual error to reraise}}
    # expected-note @below {{provide an error to raise or place 'raise'statement inside an except region}}
    raise

# expected-note @below {{or mark surrounding function as 'raises'}}
fn cannotRaise(err: Error):
    # expected-error @below {{cannot raise error in this context}}
    # expected-note @below {{try surrounding 'raise' in a 'try' block}}
    raise err

# Issue #12358
fn raise_bad_type() raises:
    raise 42  # expected-error {{cannot implicitly convert 'IntLiteral' value to 'Error' in raised value}}

# https://github.com/modularml/mojo/issues/1230
# Parser crashes on incomplete decorator
@ # expected-error {{missing decorator expression after '@'}}
fn m # expected-error {{expected '(' for argument list}}
#expected-error @-1 {{expected body statements; use 'pass' if none is required}}

# Issue #6909
# expected-error @below {{expected name for 'alias' declaration}}
# expected-note @below {{escape keyword 'True' with backticks to use it as an identifier}}
alias True = 42
