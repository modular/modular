# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 1 2 3 4 | FileCheck %s

from sys import argv


trait Coord(ImplicitlyCopyable):
    fn prettyPrint(self):
        ...


@fieldwise_init
struct Cartesian(Coord):
    var x: Int
    var y: Int

    fn prettyPrint(self):
        print("Cart:", self.x, ",", self.y)


@fieldwise_init
struct Sphere(Coord):
    var theta: Int
    var phi: Int

    fn prettyPrint(self):
        print("sphere:", self.theta, ",", self.phi)


# ===----------------------------------------------------------------------=== #
# Captured Param From Struct
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DefinesParam[T: Coord, R: Coord]:
    var state: Self.T

    fn method[C: fn(arg: Self.T) unified -> Self.R](self, impl: C) -> Self.R:
        return impl(self.state)


fn testCapturedParamFromStruct[T: Coord, R: Coord](t: T, r: R):
    fn closureImpl(arg: T) unified {var} -> R:
        t.prettyPrint()
        return r

    var definesParam = DefinesParam[T, R](t)
    _ = definesParam.method(closureImpl)


# ===----------------------------------------------------------------------=== #
# Captured Param From Function
# ===----------------------------------------------------------------------=== #


fn func[T: Coord, R: Coord, C: fn(arg: T) unified -> R](impl: C, state: T) -> R:
    return impl(state)


fn testCapturedParamFromFn[T: Coord, R: Coord](t: T, r: R):
    fn closureImpl(arg: T) unified {var} -> R:
        t.prettyPrint()
        return r

    _ = func[T, R](closureImpl, t)


# ===----------------------------------------------------------------------=== #
# Captured Param With Existing Params
# ===----------------------------------------------------------------------=== #


fn hasParam[
    R: Coord, C: fn[TT: Coord](arg: TT) unified -> R
](impl: C, x: Int) -> R:
    var state = Sphere(x, x)
    return impl(state)


fn testCapturedParamWithOtherParamsFromFn[R: Coord](t: Int, r: R):
    fn closureImpl[TT: Coord](arg: TT) unified {var} -> R:
        arg.prettyPrint()
        return r

    _ = hasParam[R, type_of(closureImpl)](closureImpl, t)


# ===----------------------------------------------------------------------=== #
# Captured Param Default
# ===----------------------------------------------------------------------=== #
fn funcWithDefault[R: Coord, C: fn[N: Int = 3]() unified -> R](impl: C) -> R:
    return impl()


fn testCapturedParamFromFnWithDefault[R: Coord](r: R):
    fn closureImpl[N: Int = 3]() unified {var} -> R:
        print(N)
        return r

    _ = funcWithDefault[R, type_of(closureImpl)](closureImpl)


# ===----------------------------------------------------------------------=== #
# Captured Param From Nested Closure
# ===----------------------------------------------------------------------=== #


fn testNestedClosureCapture[RR: Coord](r: RR, x: Int):
    fn l1[R: Coord](arg0: R) unified {var} -> R:
        fn l2[TT: Coord](arg: TT) unified {var} -> R:
            arg.prettyPrint()
            return arg0

        return hasParam[R, type_of(l2)](l2, x)

    _ = l1[RR](r)


def main():
    var one = atol(argv()[1])
    var two = atol(argv()[2])
    var three = atol(argv()[3])
    var four = atol(argv()[4])
    var x = Cartesian(one, two)
    var y = Sphere(three, four)

    # CHECK: Cart: 1 , 2
    testCapturedParamFromStruct(x, y)
    # CHECK: Cart: 1 , 2
    testCapturedParamFromFn(x, y)
    # CHECK: sphere: 4 , 4
    testCapturedParamWithOtherParamsFromFn(four, y)
    # CHECK: 3
    testCapturedParamFromFnWithDefault(x)
    # CHECK: sphere: 1 , 1
    testNestedClosureCapture(x, one)
