# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s

# ===----------------------------------------------------------------------=== #
# Stubs to allow testing without builtins
# ===----------------------------------------------------------------------=== #

alias AnyRegType = __mlir_type.`!kgen.type`
alias Float = __mlir_type.`!pop.scalar<f64>`
alias Int = __mlir_type.index


trait AnyType:
    fn __del__(owned self, /):
        ...


trait Copyable:
    fn __copyinit__(inout self, existing: Self, /):
        ...


# ===----------------------------------------------------------------------=== #
# Actual tests
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct MyInt:
    var value: Int

    @always_inline("nodebug")
    fn __init__(v: Int) -> Self:
        return Self {value: v}


fn overloaded_param[a: Int, b: MyInt]():
    pass


fn overloaded_param[a: Int, b: Int]():
    pass


# CHECK-LABEL: lit.func @"test_kw_params_overload{{.*}}"<x, y>
fn test_kw_params_overload[x: Int, y: Int]():
    # CHECK: call {{.*}}@"overloaded_param{{.*}}"<x, y>()
    overloaded_param[b=y, a=x]()

    # CHECK: call {{.*}}@"overloaded_param{{.*}}"<x, :!MyInt apply(
    # CHECK-SAME :!lit.signature<("v": index borrow) -> !MyInt> {{.*}}@MyInt::@"__init__{{.*}}", y)>()
    overloaded_param[b = MyInt(y), a=x]()


fn overloaded_arg(a: Int, b: MyInt):
    pass


fn overloaded_arg(a: Int, b: Int):
    pass


# CHECK-LABEL: lit.func @"test_kw_args_overload{{.*}}"(%x: index borrow, %y: index borrow)
fn test_kw_args_overload(x: Int, y: Int):
    # CHECK: call {{.*}}@"overloaded_arg{{.*}}"(%x, %y)
    overloaded_arg(b=y, a=x)

    # CHECK: %[[Y:.*]] = lit.call {{.*}}@MyInt::@"__init__{{.*}}"(%y)
    # CHECK-NEXT: call {{.*}}@"overloaded_arg{{.*}}"(%x, %[[Y]])
    overloaded_arg(b=MyInt(y), a=x)


# COM: test parametric overload in the presence of keyword operands.
fn take_kw_param_infer[A: AnyRegType, B: AnyRegType](a: A, b: B):
    pass


fn take_kw_param_infer[B: AnyRegType](a: MyInt, b: B):
    pass


# CHECK-LABEL: lit.func @"test_kw_args_param_infer
fn test_kw_args_param_infer(x: Int, f: Float, s: MyInt):
    # CHECK: call {{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:type index, :type scalar<f64>>(%x, %f)
    take_kw_param_infer(x, b=f)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:type index, :type scalar<f64>>(%x, %f)
    take_kw_param_infer[Int](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyRegType,AnyRegType]{{.*}}"<:type index, :type scalar<f64>>(%x, %f)
    take_kw_param_infer[Int, Float](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyRegType]{{.*}}<:type index>(%s, %x)
    take_kw_param_infer(s, b=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[AnyRegType]{{.*}}<:type index>(%s, %x)
    take_kw_param_infer(b=x, a=s)


# COM: Test overloading precedence in the presence of static methods.
struct StaticOverloadStruct:
    fn __init__(inout self):
        pass

    fn foo(inout self):
        pass

    @staticmethod
    fn foo():
        pass


# CHECK-LABEL: lit.func @"test_static_overload()"
fn test_static_overload():
    var a = StaticOverloadStruct()
    # CHECK-NEXT: %a = lit.varlet.decl
    # CHECK-NEXT: lit.call{{.*}}__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call @{{.*}}foo{{.*}}(%a)
    a.foo()


# COM: Issue https://github.com/modularml/mojo/issues/1408
# COM: Test that the number of implicit conversions is more important than
# COM: convention mismatches.
@register_passable("trivial")
struct MyElement(Copyable):
    pass


struct ConvertibleFromInt:
    fn __init__(inout self, a: Int):
        pass


struct MyContainer[T: Copyable]:
    var v: T

    fn foo(self, limits: ConvertibleFromInt):
        pass

    fn foo(self, index: Int) -> T:
        return self.v


# CHECK-LABEL: lit.func @"test_impl
fn test_impl(a: MyContainer[MyElement], b: Int):
    # CHECK: lit.call @{{.*}}@MyContainer::@"foo{{.*}}, "index": index borrow) -> !kgen.none
    _ = a.foo(b)
