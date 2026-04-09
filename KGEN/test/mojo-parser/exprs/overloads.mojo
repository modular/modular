# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated %s | FileCheck %s


# CHECK: [[SCALARF64:.*]] = #kgen.type<!lit.struct<#MLIRType <:non_struct_type scalar<f64>>>, scalar<f64>> : !TrivialRegisterPassable


struct MyInt(TrivialRegisterPassable):
    var value: Int

    @always_inline("nodebug")
    @implicit
    def __init__(out self, v: Int):
        self.value = v


def overloaded_arg(a: Int, b: MyInt):
    pass


def overloaded_arg(a: Int, b: Int):
    pass


# CHECK-LABEL: lit.fn @"test_kw_args_overload{{.*}}"(%x: !Int, %y: !Int)
def test_kw_args_overload(x: Int, y: Int):
    # CHECK: call {{.*}}@"overloaded_arg{{.*}}"(%x, %y)
    overloaded_arg(b=y, a=x)

    # CHECK: [[Y:%.*]] = lit.call {{.*}}@MyInt::@"__init__{{.*}}(%y)
    # CHECK-NEXT: call {{.*}}@"overloaded_arg{{.*}}"(%x, [[Y]])
    overloaded_arg(b=MyInt(y), a=x)


# COM: test parametric overload in the presence of keyword operands.
def take_kw_param_infer[
    A: TrivialRegisterPassable, B: TrivialRegisterPassable
](a: A, b: B):
    pass


def take_kw_param_infer[B: TrivialRegisterPassable](a: MyInt, b: B):
    pass


# CHECK-LABEL: lit.fn @"test_kw_args_param_infer
def test_kw_args_param_infer(
    x: Int, f: __mlir_type.`!pop.scalar<f64>`, s: MyInt
):
    # CHECK: call {{.*}}@"take_kw_param_infer[::TrivialRegisterPassable,::TrivialRegisterPassable]{{.*}}"<:{{.*}}!Int, {{.*}}[[SCALARF64]]>(%x, %f)
    take_kw_param_infer(x, b=f)

    # CHECK: call {{.*}}@"take_kw_param_infer[::TrivialRegisterPassable,::TrivialRegisterPassable]{{.*}}"<:{{.*}}!Int, {{.*}}[[SCALARF64]]>(%x, %f)
    take_kw_param_infer[Int](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[::TrivialRegisterPassable,::TrivialRegisterPassable]{{.*}}"<:{{.*}}!Int, {{.*}}[[SCALARF64]]>(%x, %f)
    take_kw_param_infer[Int, __mlir_type.`!pop.scalar<f64>`](b=f, a=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[::TrivialRegisterPassable]{{.*}}<:{{.*}}!Int>(%s, %x)
    take_kw_param_infer(s, b=x)

    # CHECK: call {{.*}}@"take_kw_param_infer[::TrivialRegisterPassable]{{.*}}<:{{.*}}!Int>(%s, %x)
    take_kw_param_infer(b=x, a=s)


# COM: Test overloading precedence in the presence of static methods.
struct StaticOverloadStruct:
    def __init__(out self):
        pass

    def foo(mut self):
        pass

    @staticmethod
    def foo():
        pass


# CHECK-LABEL: lit.fn @"test_static_overload()"
def test_static_overload():
    var a = StaticOverloadStruct()
    # CHECK-NEXT: %a = lit.var.decl
    # CHECK-NEXT: lit.call{{.*}}__init__{{.*}}(%a)
    # CHECK-NEXT: lit.call {{.*}}foo{{.*}}(%a)
    a.foo()


# COM: Issue https://github.com/modular/mojo/issues/1408
# COM: Test that the number of implicit conversions is more important than
# COM: convention mismatches.
struct MyElement(TrivialRegisterPassable):
    pass


struct ConvertibleFromInt:
    @implicit
    def __init__(out self, a: Int):
        pass


struct MyContainer[T: ImplicitlyCopyable]:
    var v: Self.T

    def foo(self, limits: ConvertibleFromInt):
        pass

    def foo(self, index: Int) -> Self.T:
        return self.v


# CHECK-LABEL: lit.fn @"test_impl
def test_impl(a: MyContainer[MyElement], b: Int):
    # CHECK: lit.call {{.*}}@MyContainer::@"foo{{.*}}, "index": !Int
    _ = a.foo(b)
