# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


fn take_func_without_arg_name[f: fn (Int) -> None]():
    pass


fn func_with_arg_name(a: Int):
    pass


# COM: Issue https://github.com/modular/mojo/issues/1307
# COM: Test that functions with defaults can be passed where no defaults are expected
fn take_func_without_default[f: fn (a: Int) -> None]():
    pass


fn func_with_default(a: Int = 0):
    pass


# CHECK-LABEL: lit.fn @"test_passing_funcs
fn test_passing_funcs():
    # CHECK: lit.call tail @{{.*}}::@"take_func_without_arg_name{{.*}}"<
    # CHECK-SAME: :!lit.generator<(!Int, |) -> !kgen.none> rebind(:!lit.generator<("a": !Int) -> !kgen.none>
    take_func_without_arg_name[func_with_arg_name]()

    # CHECK: lit.call {{.*}}::@"take_func_without_default{{.*}}"<
    # CHECK-SAME: :!lit.generator<("a": !Int) -> !kgen.none> rebind(:!lit.generator<("a": !Int = {0}) -> !kgen.none>
    take_func_without_default[func_with_default]()


fn fn_doesnt_raise() -> Int: pass
fn fn_returns_ref(x: String) -> ref [x] String: pass

# CHECK-LABEL: lit.fn @"test_more_conversions
fn test_more_conversions():
  # CHECK: %test_raises = lit.var.decl
  # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
  # CHECK-NEXT: lit.ref.store [[TMP]], %test_raises
  var test_raises : fn () raises -> Int = fn_doesnt_raise

  # CHECK: %test_result_convert = lit.var.decl
  # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
  # CHECK-NEXT: lit.ref.store [[TMP]], %test_result_convert
  var test_result_convert : fn () raises -> Float32 = fn_doesnt_raise

  # CHECK: %test_error_convert = lit.var.decl
  # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
  # CHECK-NEXT: lit.ref.store [[TMP]], %test_error_convert
  var test_error_convert : fn () raises Float32 -> Float32 = fn_doesnt_raise

  # CHECK: %test_ref_result_convert = lit.var.decl
  # CHECK-NEXT: [[TMP:%.*]] = kgen.create_closure
  # CHECK-NEXT: lit.ref.store [[TMP]], %test_ref_result_convert
  var test_ref_result_convert : fn (x: String) -> String = fn_returns_ref


# Check that we can take /explicitly copyable/ return values as ref returns.
# This is a hack (see EXPLICIT-COPY-REF-RETURN) to support __next__ promoting
# its result type.  We should remove this when we have more powerful Iterator
# traits and origins that can support that.
trait TraitExpectingValueReturn:
    comptime Element: ImplicitlyDestructible
    fn return_value(self) -> Self.Element:
        ...
struct StructProvidingRefReturn[T: Copyable](TraitExpectingValueReturn):
    comptime Element = Self.T
    fn return_value(self) -> ref [self] Self.T:
        pass


struct FromType:
    var n: Int
    fn __init__(out self, n: Int):
        self.n = n

struct ToTypeImm:
    var n: Int
    @implicit
    fn __init__[O: ImmutOrigin](out self, ref [O]f: FromType):
        self.n = f.n

struct ToTypeMut:
    var n: Int

    @implicit
    fn __init__[O: MutOrigin](out self, ref [O]f: FromType):
        self.n = f.n

fn useToType(s: ToTypeImm): pass
fn useToType(s: ToTypeMut): pass

# Tests that implicit conversions that are ref-dependent are cached correctly.
# If the implicit conversion cache does not take into account the RefType, the
# second call to useToType will emit an error.
fn test[O: MutOrigin, O2: ImmutOrigin](ref [O]fImm: FromType, ref [O2]fMut: FromType):
    # CHECK: lit.call {{.*}}@ToTypeMut::@"__init__
    useToType(fImm)
    # CHECK: lit.call {{.*}}@ToTypeImm::@"__init__
    useToType(fMut)

fn main():
    f1 = FromType(1)
    f2 = FromType(2)
    test(f1, f2)
