# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s


trait Barable:
    fn bar(self):
        ...


trait Foo:
    fn rp_return_params[T: Int](self) -> Int:
        pass

    @staticmethod
    fn rp_return_static_multis(x: Int, y: Int) -> Int:
        pass

    fn rp_return_multi_params[T: Int, T2: Int](self, x: Int) -> Int:
        pass

    fn ref_return(self) -> String:
        pass

    fn ref_return_params[T: Int](self, name: String) -> String:
        pass

    fn rp_return_raises(self) raises -> Int:
        pass

    fn ref_return_raises(self) raises -> String:
        pass

    fn parametric_ref_args(self, ref x: Int, ref y: Int) -> Int:
        pass


trait FooActual(Absable, Foo, Intable):
    comptime P: Int

    fn rp_return_params[T: Int](self) -> Int:
        var res = (
            Int(self.__abs__())
            + T
            + Self.P
            + Self.rp_return_static_multis(T, T)
        )
        return res

    @staticmethod
    fn rp_return_static_multis(x: Int, y: Int) -> Int:
        return x + y + Self.P

    fn rp_return_multi_params[T: Int, T2: Int](self, x: Int) -> Int:
        print("In FooActual.rp_return_multi_params")
        var res = Int(self.__abs__()) + T + Self.P + T2 + x
        return res

    fn ref_return(self) -> String:
        return "FooActual ref_return value: " + String(Int(self.__abs__()))

    fn ref_return_params[T: Int](self, name: String) -> String:
        return (
            "FooActual ref_return_params: Hello "
            + name
            + ", params T="
            + String(T)
            + ", alias P="
            + String(Self.P)
        )

    fn rp_return_raises(self) raises -> Int:
        var val = Int(self.__abs__())
        if val <= 0:
            raise Error(
                "rp_return_raises: Value must be positive, got: " + String(val)
            )
        return val

    fn ref_return_raises(self) raises -> String:
        var val = Int(self.__abs__())
        if val == 0:
            raise Error(
                "ref_return_raises: Cannot describe object with zero value"
            )
        if val < 0:
            return "ref_return_raises: Negative value with absolute: " + String(
                val
            )
        else:
            return "ref_return_raises: Positive value: " + String(val)

    fn parametric_ref_args(self, ref x: Int, ref y: Int) -> Int:
        return x + y


@fieldwise_init
struct Bar(Barable, FooActual):
    var x: Int
    comptime P: Int = 10

    fn __int__(self) -> Int:
        return self.x

    fn __abs__(self) -> Self:
        return Self(abs(self.x))

    fn bar(self):
        print("In Bar implementation, called bar()")

    fn rp_return_multi_params[T: Int, T2: Int](self, x: Int) -> Int:
        print("In Bar.rp_return_multi_params")
        var res = Int(self.__abs__()) + T + Self.P - T2 - x
        return res


fn generic_trait_caller[T: Foo](x: T):
    print("In generic_trait_caller[T: Foo](x: T) -> Int")
    print(
        "x.rp_return_multi_params[20, 10](10) =",
        x.rp_return_multi_params[20, 10](10),
    )


trait AATrait1:
    comptime X: ImplicitlyCopyable

    fn zork(self, x: Self.X) -> Self.X:
        print("In AATrait1.zork")
        return x


trait AATrait2:
    comptime X: ImplicitlyCopyable & Movable

    fn zork(self, x: Self.X) -> Self.X:
        print("In AATrait2.zork")
        return x


@fieldwise_init
struct AAStruct(AATrait1, AATrait2):
    comptime X = Int

    fn zork(self, x: Self.X) -> Self.X:
        print("In Foo.zork")
        return x


@fieldwise_init
@register_passable("trivial")
struct ParamRP[x: Int, y: Int]:
    var z: Int


trait ParamTraitWithParameterizedInputs:
    @staticmethod
    fn process_parameterized[T: Barable](item: T) -> Int:
        item.bar()
        return 42

    fn return_parameterized[x: Int, y: Int](self) -> ParamRP[x, y]:
        return ParamRP[x, y](x + y)


@fieldwise_init
struct ParamTestStruct(ParamTraitWithParameterizedInputs):
    pass


@fieldwise_init
struct Bag[x: Int]:
    var v: Int


trait NonRPParamDefaultTrait:
    fn sum_bag[x: Int](self, bag: Bag[x]) -> Int:
        return bag.v + x


@fieldwise_init
struct NonRPStruct(NonRPParamDefaultTrait):
    pass


def main():
    var b = Bar(-20)

    # CHECK: b.rp_return_params[10]() = 70
    print(
        "b.rp_return_params[10]() =",
        b.rp_return_params[10](),
        end="\n\n",
    )
    # CHECK: Bar.rp_return_static_multis(10, 5) = 25
    print(
        "Bar.rp_return_static_multis(10, 5) =",
        Bar.rp_return_static_multis(10, 5),
        end="\n\n",
    )
    # CHECK: In Bar.rp_return_multi_params
    # CHECK-NEXT: b.rp_return_multi_params[10, 20](30) = -10
    print(
        "b.rp_return_multi_params[10, 20](30) =",
        b.rp_return_multi_params[10, 20](30),
        end="\n\n",
    )
    # CHECK: In generic_trait_caller[T: Foo](x: T) -> Int
    # CHECK-NEXT: In Bar.rp_return_multi_params
    # CHECK-NEXT: x.rp_return_multi_params[20, 10](10) = 30
    generic_trait_caller(b)

    # CHECK: b.parametric_ref_args(10, 20) = 30
    print(
        "b.parametric_ref_args(10, 20) =",
        b.parametric_ref_args(10, 20),
        end="\n\n",
    )

    # Test string-returning functions
    # CHECK: FooActual ref_return value: 20
    print(b.ref_return())
    # CHECK: FooActual ref_return_params: Hello Alice, params T=5, alias P=10
    print(b.ref_return_params[5]("Alice"))

    # Test raising functions - these should succeed
    try:
        # CHECK: rp_return_raises() = 20
        print("rp_return_raises() =", b.rp_return_raises())
        # CHECK: ref_return_raises: Positive value: 20
        print(b.ref_return_raises())
    except e:
        print("Unexpected error:", e)

    # Test raising functions - these should fail
    var b_zero = Bar(0)
    try:
        _ = b_zero.rp_return_raises()
        print("Should not reach here")
    except e:
        # CHECK: Caught expected error: rp_return_raises: Value must be positive, got: 0
        print("Caught expected error:", e)

    try:
        _ = b_zero.ref_return_raises()
        print("Should not reach here")
    except e:
        # CHECK: Caught expected error: ref_return_raises: Cannot describe object with zero value
        print("Caught expected error:", e)

    # Test negative value case for string-returning raising function
    var b_negative = Bar(-5)
    try:
        # CHECK: ref_return_raises: Positive value: 5
        print(b_negative.ref_return_raises())
    except e:
        print("Unexpected error:", e)

    # Test that we handle trait methods that use associated aliases correctly.
    var a = AAStruct()

    # CHECK: In Foo.zork
    # CHECK-NEXT: 10
    print(a.zork(10))

    # Test parameterized default trait methods
    var param_test = ParamTestStruct()

    # Test default trait method that takes parameterized type as input
    var bar_for_param = Bar(5)
    # CHECK: In Bar implementation, called bar()
    # CHECK-NEXT: 42
    print(ParamTestStruct.process_parameterized(bar_for_param))

    # Test default trait method that returns parameterized register passable trivial type
    # CHECK: 15
    var result = param_test.return_parameterized[10, 5]()
    print(result.z)

    # Test default trait method that takes non-register-passable parameterized type
    var default_struct = NonRPStruct()
    var bag = Bag[22](3)
    # CHECK: 25
    print(default_struct.sum_bag[22](bag))
