# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

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
  fn __copyinit__(out self, existing: Self):
    pass

def test_bool_context(a: NotBoolConvertible):
  if a: # expected-error {{NotBoolConvertible' does not implement the '__bool__' method}}
     pass

fn test_if_decorator(a: Bool):
  @not_good() # expected-error {{unsupported decorator on 'if' statement}}
  if a:
    pass

  @parameter
  if 1:
    pass
  elif a:  # expected-error {{cannot use a dynamic value in '@parameter if' condition}}
    pass

##===----------------------------------------------------------------------===##
# For
##===----------------------------------------------------------------------===##

struct my_iter_no_len:
    fn __init__(out self): pass
    fn __next__(mut self) -> Int: return 0


struct MyList_range_no_len:
    fn __init__(out self): pass
    fn __iter__(self) -> my_iter_no_len: return my_iter_no_len()


struct my_iter_no_next:
    fn __init__(out self): pass
    fn __has_next__(self) -> Bool: return False


struct MyList_range_no_next:
    fn __init__(out self): pass
    fn __iter__(self) -> my_iter_no_next: return my_iter_no_next()


struct MyList_no_iter:
    fn __init__(out self): pass

@value
struct MyFloat:
    pass

struct my_iter_wrong_int:
    fn __init__(out self): pass
    fn __next__(mut self) -> Int: return 0
    fn __has_next__(self: my_iter_wrong_int) -> MyFloat: return MyFloat()


struct MyList_invalid_boxed_type:
    fn __init__(out self): pass
    fn __iter__(self) -> my_iter_wrong_int: return my_iter_wrong_int()


fn test():
    var my_list_no_len = MyList_range_no_len()
    var my_list_no_next = MyList_range_no_next()
    var my_list_no_iter = MyList_no_iter()
    var my_list_invalid_int = MyList_invalid_boxed_type()

    # expected-error @+1 {{'my_iter_no_len' does not implement the '__has_next__' method}}
    for item in my_list_no_len:
        pass

    # expected-error @+1 {{'my_iter_no_next' does not implement the '__next__' method}}
    for item in my_list_no_next:
        pass

    # expected-error @+1 {{'MyList_no_iter' does not implement the '__iter__' method}}
    for item in my_list_no_iter:
        pass

    # expected-error @+1 {{'MyFloat' does not implement the '__bool__' method}}
    for item in my_list_invalid_int:
        pass

    # expected-error @+1 {{'my_iter_no_next' does not implement the '__next__' method}}
    for key, item in my_list_no_next:
        pass

# Issue #18599
fn spurious_for_loop_variable_unknown_decl():
  # expected-error @below {{'FloatLiteral[1]' does not implement the '__iter__' method}}
  for i in 1.0:
    # Note that the bug in issue #18599 is that after the above error, another error
    # will be spuriously raised about i not being bound.  So the real check in
    # this test is that no further error is raised.
    _ = i


struct ListValueInt:
    fn __init__(out self): pass
    fn __iter__(self) -> ListValueInt: return ListValueInt()
    fn __next__(mut self) -> Int: return 0
    fn __has_next__(self) -> Bool: return False

struct ListValueStringRef:
    fn __init__(out self): pass
    fn __iter__(self) -> ListValueStringRef: return ListValueStringRef()
    fn __next__(mut self) -> ref [self] String: pass
    fn __has_next__(self) -> Bool: return False


def loop_variable_scoped():
  for i in ListValueInt(): pass
  _ = i # expected-error {{use of unknown declaration 'i'}}

  for elt in ListValueStringRef():
    elt = "foo" # expected-error {{expression must be mutable in assignment}}

##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##

struct ExampleCM:
  fn __moveinit__(out self, owned other: Self): pass
  fn __enter__(self) -> Int:
    return 42
  fn __exit__(self):
    pass # normal
  fn __exit__(self, err: Error) -> Bool:
    return True # Raise

def withUsingImmutableVariable(owned a: ExampleCM):
  var x = 77
  with a^ as x:
    pass

# External Issue #529 https://github.com/modular/mojo/issues/529
def withWithNoColon(owned a: ExampleCM):
  # expected-error @below {{expected ':' or ',' after 'with' expression}}
  with a^ as b

fn withNoRaise(owned mgr: ExampleCM): # expected-note {{or mark surrounding function as 'raises'}}
  with mgr^:
    # expected-error @below {{cannot raise error in this context}}
    # expected-note @below {{try surrounding 'raise' in a 'try' block}}
    raise Error()

  # Allow try-finally, but in a non-raising region.
  try:
    # expected-error @below {{cannot raise error in this context}}
    # expected-note @below {{try surrounding 'raise' in a 'try' block}}
    raise Error()
  finally:
    pass

# Poor error when with context managers that take ownership in enter
# https://github.com/modularml/modular/issues/23100
struct BadCM: # expected-note {{'BadCM' declared here}}
  fn __init__(out self): pass

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
