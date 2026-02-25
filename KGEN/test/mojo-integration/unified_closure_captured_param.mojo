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


trait Euclidean:
    fn distance(self, other: Self) -> Int:
        ...


@fieldwise_init
struct Cartesian(Coord, Euclidean):
    var x: Int
    var y: Int

    fn prettyPrint(self):
        print("Cart:", self.x, ",", self.y)

    fn distance(self, other: Cartesian) -> Int:
        return self.x - other.x


@fieldwise_init
struct Sphere(Coord):
    var theta: Int
    var phi: Int

    fn prettyPrint(self):
        print("sphere:", self.theta, ",", self.phi)


@fieldwise_init
struct Polar(Coord):
    var r: Int
    var theta: Int

    fn prettyPrint(self):
        print("polar:", self.r, ",", self.theta)


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


fn topLevelConcrete[TT: Coord](arg: TT) -> Cartesian:
    arg.prettyPrint()
    return Cartesian(6, 7)


fn testTopLevelConcreteWithOtherParams(t: Int):
    var result = hasParam[Cartesian, topLevelConcrete](topLevelConcrete, t)
    result.prettyPrint()


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


# ===----------------------------------------------------------------------=== #
# Lazy Conformance
# ===----------------------------------------------------------------------=== #


fn testLazyConformance[NOT_T: Coord](something: NOT_T):
    fn closureImpl(arg1: NOT_T) unified {var} -> Sphere:
        something.prettyPrint()
        return Sphere(33, 34)

    var definesParam = DefinesParam[NOT_T, Sphere](something)
    _ = definesParam.method(closureImpl)


fn manyCaptures[
    A: Coord, B: Coord, D: Coord, F: fn[C: Coord](a: A, b: B, c: C) unified -> D
](impl: F, arg1: A, arg2: B, r: Int):
    var polar = Polar(r, r)
    var result = impl(arg1, arg2, polar)
    result.prettyPrint()


fn testLazyConformanceManyCaptures[BB: Coord](arg: BB, a0: Cartesian, r: Int):
    fn closure[CC: Coord](a1: Cartesian, b1: BB, c1: CC) unified {var} -> BB:
        a1.prettyPrint()
        b1.prettyPrint()
        return arg

    manyCaptures[Cartesian, BB, BB, type_of(closure)](closure, a0, arg, r)


fn superset[B: Coord, D: Coord, F: fn(b: B) unified -> D](impl: F, arg: B):
    var result = impl(arg)
    result.prettyPrint()


fn testLazyConformanceSuperset[BB: Coord & Euclidean](arg: BB):
    fn closure(b1: BB) unified {var} -> BB:
        return arg

    superset[BB, BB, type_of(closure)](closure, arg)


def main():
    var one = atol(argv()[1])
    var two = atol(argv()[2])
    var three = atol(argv()[3])
    var four = atol(argv()[4])
    var x = Cartesian(one, two)
    var y = Sphere(three, four)
    var polar = Polar(two, two)

    # CHECK: Cart: 1 , 2
    testCapturedParamFromStruct(x, y)
    # CHECK: Cart: 1 , 2
    testCapturedParamFromFn(x, y)
    # CHECK: sphere: 4 , 4
    testCapturedParamWithOtherParamsFromFn(four, y)
    # CHECK: sphere: 3 , 3
    # CHECK: Cart: 6 , 7
    testTopLevelConcreteWithOtherParams(three)
    # CHECK: 3
    testCapturedParamFromFnWithDefault(x)
    # CHECK: sphere: 1 , 1
    testNestedClosureCapture(x, one)
    # CHECK: Cart: 1 , 2
    testLazyConformance(x)
    # CHECK: Cart: 1 , 2
    # CHECK: polar: 2 , 2
    # CHECK: polar: 2 , 2
    testLazyConformanceManyCaptures(polar, x, one)
    # CHECK: Cart: 1 , 2
    testLazyConformanceSuperset(x)
