# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


@register_passable("trivial")
struct MyInt:
    var value: Index

    @always_inline("nodebug")
    @implicit
    fn __init__(out self, v: Index):
        self.value = v


fn overloaded_param[a: Index, b: MyInt]():
    pass


fn overloaded_param[a: Index, b: Index]():
    pass


# CHECK-LABEL: lit.fn @"test_kw_params_overload{{.*}}"<x, y>
fn test_kw_params_overload[x: Index, y: Index]():
    # CHECK: call {{.*}}@"overloaded_param{{.*}}"<x, y>()
    overloaded_param[b=y, a=x]()

    # CHECK: call {{.*}}@"overloaded_param{{.*}}"<x, :!MyInt apply(
    # CHECK-SAME :!lit.generator<("v": index) -> !MyInt> {{.*}}@MyInt::@"__init__{{.*}}", y)>()
    overloaded_param[b = MyInt(y), a=x]()


fn overloaded_arg(a: Index, b: MyInt):
    pass


fn overloaded_arg(a: Index, b: Index):
    pass


# CHECK-LABEL: lit.fn @"test_kw_args_overload{{.*}}"(%x: index, %y: index)
fn test_kw_args_overload(x: Index, y: Index):
    # CHECK: call {{.*}}@"overloaded_arg{{.*}}"(%x, %y)
    overloaded_arg(b=y, a=x)

    # CHECK: [[Y:%.*]] = lit.call {{.*}}@MyInt::@"__init__{{.*}}(%y)
    # CHECK-NEXT: call {{.*}}@"overloaded_arg{{.*}}"(%x, [[Y]])
    overloaded_arg(b=MyInt(y), a=x)


# COM: test parametric overload in the presence of keyword operands.
fn take_kw_param_infer[A: AnyTrivialRegType, B: AnyTrivialRegType](a: A, b: B):
    pass


fn take_kw_param_infer[B: AnyTrivialRegType](a: MyInt, b: B):
    pass


# CHECK-LABEL: lit.fn @"test_kw_args_param_infer
fn test_kw_args_param_infer(x: Index, f: float, s: MyInt):
    # CHECK: call {{.*}}@"take_kw_param_infer[AnyTrivialRegType,AnyTrivialRegType]{{.*}}"<:type index, :type scalar<f64>>(%x, %f)
    take_kw_param_infer(x, b=f)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyTrivialRegType,AnyTrivialRegType]{{.*}}"<:type index, :type scalar<f64>>(%x, %f)
    take_kw_param_infer[Index](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyTrivialRegType,AnyTrivialRegType]{{.*}}"<:type index, :type scalar<f64>>(%x, %f)
    take_kw_param_infer[Index, float](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyTrivialRegType]{{.*}}<:type index>(%s, %x)
    take_kw_param_infer(s, b=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyTrivialRegType]{{.*}}<:type index>(%s, %x)
    take_kw_param_infer(b=x, a=s)


# COM: Test overloading precedence in the presence of static methods.
struct StaticOverloadStruct:
    fn __init__(out self):
        pass

    fn foo(mut self):
        pass

    @staticmethod
    fn foo():
        pass


# CHECK-LABEL: lit.fn @"test_static_overload()"
fn test_static_overload():
    var a = StaticOverloadStruct()
    # CHECK-NEXT: %a = lit.var.decl
    # CHECK-NEXT: lit.call{{.*}}__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call @{{.*}}foo{{.*}}(%a)
    a.foo()


# COM: Issue https://github.com/modular/mojo/issues/1408
# COM: Test that the number of implicit conversions is more important than
# COM: convention mismatches.
@register_passable("trivial")
struct MyElement(Copyable):
    pass


struct ConvertibleFromInt:
    @implicit
    fn __init__(out self, a: Index):
        pass


struct MyContainer[T: Copyable]:
    var v: T

    fn foo(self, limits: ConvertibleFromInt):
        pass

    fn foo(self, index: Index) -> T:
        return self.v


# CHECK-LABEL: lit.fn @"test_impl
fn test_impl(a: MyContainer[MyElement], b: Index):
    # CHECK: lit.call @{{.*}}@MyContainer::@"foo{{.*}}, "index": index
    _ = a.foo(b)
