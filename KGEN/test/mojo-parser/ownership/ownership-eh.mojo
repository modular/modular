# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: kgen-translate -import-mojo %s --mlir-print-debuginfo -o %t.mlir
# RUN: kgen-opt %t.mlir -lower-semantic-cf -check-lifetimes -verify-diagnostics | FileCheck %s
# RUN: kgen-translate -import-mojo %s --mlir-print-debuginfo --debug-level full -o /dev/null

# Error Handling related CheckLifetimes tests.


# CHECK-LABEL: lit.struct.decl @RegExample
# CHECK: attributes {{.*}}destructor = #kgen.symbol.constant<{{.*}}@RegExample::@"__del__
@register_passable
struct RegExample:
  fn __init__() -> Self:
    return RegExample{}
  fn __copyinit__(self) -> Self: # CHECK: lit.func @"__copyinit__
    return RegExample{}

  # Test a raising constructor.
  # CHECK-LABEL: lit.func @"__init__{{.*}}MemExample{{.*}}MemExample
  fn __init__(a: MemExample, b: MemExample) raises -> Self:
    # CHECK-NEXT: %0 = kgen.param.materialize: !RegExample
    # CHECK-NEXT: %1 = kgen.variant.create %0
    # CHECK-NEXT: kgen.return %1
    return RegExample{}

  fn noop(self): pass
  fn __del__(owned self): pass
  fn mutate(inout self): pass

struct MemExample:
  var x : Int
  fn __init__(inout self): self.x = 42; pass
  fn noop(self): pass
  fn __moveinit__(inout self, owned existing: Self): self.x = existing.x
  fn __copyinit__(inout self, existing: Self): self.x = existing.x
  fn __bool__(self) -> Bool: return True
  fn __del__(owned self): pass


def foo(x: Int): pass
fn use(x: Int): pass

# Use of uninitialized value after call to def function

# CHECK-LABEL: lit.func @"error_handling_int_let
# https://github.com/modularml/modular/issues/25419
def error_handling_int_let():
    # CHECK: lit.varlet.decl "x"
    var x: Int = 1
    _ = foo(x)
    use(x)

fn somethingThatRaises() raises: pass

# CHECK-LABEL: lit.func @"thing_that_raises
fn thing_that_raises(c: __mlir_type.i1) raises -> MemExample:
    # CHECK-NEXT: lit.call {{.*}}somethingThatRaises
    # CHECK-NEXT: %1 = lit.handle_variant
    # CHECK-NEXT:   %5 = kgen.variant.take
    # CHECK-NEXT:   lit.yield %5
    # CHECK-NEXT: } else {
    # CHECK-NEXT:  %5 = kgen.variant.take %0, 0
    # CHECK-NEXT:  %6 = kgen.variant.create %5, 0
    # CHECK-NEXT:  lit.error_return %6
    # CHECK-NEXT: }
    somethingThatRaises()

    # CHECK-NEXT:   hlcf.if %c {
    # CHECK-NEXT:      = lit.call {{.*}}__init__
    # CHECK-NOT:       __del__
    # CHECK: kgen.return
    if c:
        return MemExample()
    # CHECK-NEXT:  } else {
    raise Error("TypeError: cannot invert values of this type")

struct RaisingInit:
    var stream: Int
    fn __init__(inout self, flags: Int = 0) raises:
        let stream = 4
        # This can raise, but 'self' doesn't need to be initialized.
        _ = somethingThatRaises()
        self.stream = stream


fn may_throw() raises -> RegExample:
  return RegExample()

# CHECK-LABEL: lit.func @"propagate_reg_error
fn propagate_reg_error() raises:
   # CHECK-NEXT: %0 = lit.call {{.*}}may_throw
   # CHECK-NEXT: %1 = lit.handle_variant %0
   # CHECK-NEXT:   [[REG:%.*]] = kgen.variant.take %0, 1 : <!Error, !RegExample>
   # CHECK-NEXT:   lit.yield [[REG]]
   # CHECK-NEXT: } else {
   #               .. stuff ..
   # CHECK:        lit.error_return
   # CHECK-NEXT: }
   # CHECK-NEXT: %2 = lit.call {{.*}}@RegExample::@"__del__{{.*}}(%1)
   # CHECK-NEXT: kgen.param.constant: none
   # CHECK-NEXT: kgen.variant.create %none, 1
   # CHECK-NEXT: kgen.return
    _ = may_throw()


# CHECK-LABEL: lit.struct.decl @BigRegExample
@register_passable
struct BigRegExample:
  var a: RegExample
  var b: RegExample

  # Test a raising constructor.
  # CHECK-LABEL: lit.func @"__init__{{.*}}MemExample{{.*}}MemExample
  fn __init__(a: MemExample, b: MemExample) raises -> Self:
    # CHECK-NEXT: %0 = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %1 = lit.call {{.*}}__init__{{.*}}()
    # CHECK-NEXT: %2 = lit.struct.create(a=%0, b=%1)
    # CHECK-NEXT: %3 = kgen.variant.create %2, 1
    # CHECK-NEXT: kgen.return %3
    return BigRegExample{a: RegExample(), b: RegExample() }


struct MyStringReturningCtx:
    var s: String
    fn __init__(inout self):
        self.s = "hey"
    fn __enter__(owned self) -> Self:
        return self ^
    fn __moveinit__(inout self, owned existing: Self):
        self.s = existing.s
    fn read(self) raises -> String:
        return str(self.s)

# CHECK: lit.func @"testErrorReturn
fn testErrorReturn() raises:
    let input: String
    # CHECK: try
    with MyStringReturningCtx() as ctx:
        # CHECK-NOT: @MyStringReturningCtx::@"__del__
        let x = ctx.read()
        input = "hello"
    # CHECK: except
    print(input)
