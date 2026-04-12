# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# Return
##===----------------------------------------------------------------------===##

def foo() raises:
# expected-error @+1 {{unexpected token in expression}}
  return pass

return 32 # expected-error {{'return' must be inside a function; move this into a function body}}

##===----------------------------------------------------------------------===##
# If / While
##===----------------------------------------------------------------------===##

def elif_parse_error(a: Bool) raises:
  if a:
    pass
 elif a: # expected-error {{unknown tokens at the end of a declaration}}
    pass
 else:
    pass

struct NotBoolConvertible:
  def __init__(out self, *, copy: Self):
    pass

def test_bool_context(a: NotBoolConvertible) raises:
  if a: # expected-error {{NotBoolConvertible' does not implement the '__bool__' method}}
     pass

def test_if_decorator(a: Bool):
  @not_good() # expected-error {{unsupported decorator on 'if' statement}}
  if a:
    pass

  comptime if 1:
    pass
  elif a:  # expected-error {{cannot use a dynamic value in 'comptime if' condition}}
    pass

def test_comptime_if_dynamic_elif(a: Bool):
  comptime if 1:
    pass
  elif a:  # expected-error {{cannot use a dynamic value in 'comptime if' condition}}
    pass

def test_decorator_with_comptime_if():
  @parameter
  comptime if True:  # expected-error {{@parameter decorator is redundant on 'comptime if'}}
    pass

  @not_good()
  comptime if True:  # expected-error {{'comptime if' statement does not allow decorators}}
    pass

def test_comptime_elif_not_allowed(a: Bool):
  if a:
    pass
  comptime elif a:  # expected-error {{'comptime' cannot be used with 'elif'}}
    pass

##===----------------------------------------------------------------------===##
# For
##===----------------------------------------------------------------------===##

struct my_iter_no_next:
    def __init__(out self): pass


struct MyList_range_no_next:
    def __init__(out self): pass
    def __iter__(self) -> my_iter_no_next: return my_iter_no_next()


struct MyList_no_iter:
    def __init__(out self): pass

@fieldwise_init
struct MyFloat:
    pass

def test():
    var my_list_no_next = MyList_range_no_next()
    var my_list_no_iter = MyList_no_iter()


    # expected-error @+1 {{'my_iter_no_next' is not 'Iterable'; conform and add a '__next__' method to use it in a 'for' loop}}
    for item in my_list_no_next:
        pass

    # expected-error @+1 {{'MyList_no_iter' does not implement the '__iter__' method}}
    for item in my_list_no_iter:
        pass

    # expected-error @+1 {{'my_iter_no_next' is not 'Iterable'; conform and add a '__next__' method to use it in a 'for' loop}}
    for key, item in my_list_no_next:
        pass

# Issue #18599
def spurious_for_loop_variable_unknown_decl():
  # expected-error @below {{'FloatLiteral[1]' does not implement the '__iter__' method}}
  for i in 1.0:
    # Note that the bug in issue #18599 is that after the above error, another error
    # will be spuriously raised about i not being bound.  So the real check in
    # this test is that no further error is raised.
    _ = i


struct ListValueInt:
    def __init__(out self): pass
    def __iter__(self) -> ListValueInt: return ListValueInt()
    def __next__(mut self) raises StopIteration -> Int: return 0

struct ListValueStringRef:
    def __init__(out self): pass
    def __iter__(self) -> ListValueStringRef: return ListValueStringRef()
    def __next__(mut self) raises StopIteration -> ref [self] String: pass


def loop_variable_scoped() raises:
  for i in ListValueInt():
     i = i   # expected-error {{expression must be mutable in assignment}}
  _ = i # expected-error {{use of unknown declaration 'i'}}

  for elt in ListValueStringRef():
    elt = "foo" # expected-error {{expression must be mutable in assignment}}

  for ref elt in ListValueStringRef():
    elt = "foo" # ok


##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##

struct ExampleCM(Movable):
  def __enter__(self) -> Int:
    return 42
  def __exit__(self):
    pass # normal
  def __exit__(self, err: Error) -> Bool:
    return True # Raise

def withUsingImmutableVariable(var a: ExampleCM) raises:
  var x = 77
  with a^ as x:
    pass

# External Issue #529 https://github.com/modular/mojo/issues/529
def withWithNoColon(var a: ExampleCM) raises:
  # expected-error @below {{expected ':' or ',' after 'with' expression}}
  with a^ as b

def withNoRaise(var mgr: ExampleCM):
  with mgr^:
    # expected-error @below {{'raise' requires a surrounding 'try' block or the enclosing function to declare 'raises'}}
    raise Error()

  # Allow try-finally, but in a non-raising region.
  try:
    # expected-error @below {{'raise' requires a surrounding 'try' block or the enclosing function to declare 'raises'}}
    raise Error()
  finally:
    pass

# Poor error when with context managers that take ownership in enter
# https://github.com/modularml/modular/issues/23100
struct BadCM: # expected-note {{'BadCM' declared here}}
  def __init__(out self): pass

  def __enter__(var self) -> Int:
    return 42
  def __exit__(self):
    pass # normal
  def __exit__(self, err: Error) -> Bool:
    return True # Raise

def noop(a: Int): pass

def testBadCM():
  # expected-error @+1 {{context manager of type 'BadCM' defines a consuming __enter__ method as well as an __exit__ method; either remove 'var' from its '__enter__' method or remove the '__exit__' method}}
  with BadCM():
    pass


struct MyBool(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.i1

    # expected-note @below {{function declared here}}
    def __mlir_i1__(self) -> __mlir_type.i1:
        return self._mlir_value

def noIndentError():
  for i in ListValueInt():
    # expected-error @+1 {{value passed to 'self' cannot be converted from type value 'MyBool' to an instance of 'MyBool'; did you mean to instantiate 'MyBool'?}}
    if MyBool: # no error 'statements must start at the beginning of a line' should be printed
      pass

