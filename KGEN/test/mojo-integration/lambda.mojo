# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# A lambda desugars to an anonymous closure, constructed at emit time. An
# explicit capture list and return type are required (elision is not yet
# supported).


def callInt[T: def(x: Int) -> Int, //](f: T, arg: Int):
    print(f(arg))


# The return type `R` appears only inside the function-trait bound on `F`, so it
# is inferred from the lambda passed for `f` (closure type-parameter inference).
def callRet[R: AnyType, F: def(x: Int) -> R, //](f: F, arg: Int) -> R:
    return f(arg)


def addBase(base: Int):
    # A lambda capturing the enclosing function's parameter `base`. A
    # register-passable `Int` argument can only be captured by `imm`.
    callInt(lambda (x: Int) {imm base} -> Int: x + base, 5)


def addEnclosingParam[N: Int]() -> Int:
    # A lambda referencing the enclosing parametric function's compile-time
    # parameter `N`. A parameter is a compile-time value, so it is referenced
    # directly rather than through the (empty) capture list.
    return (lambda (x: Int) {} -> Int: x + N)(3)


def main():
    # Non-capturing lambda.
    callInt(lambda (x: Int) {} -> Int: x + 1, 4)
    # CHECK: 5

    # Capturing lambda: the default '{mut}' convention captures `z`.
    var z = 10
    callInt(lambda (x: Int) {mut} -> Int: x + z, 5)
    # CHECK: 15

    # `R` is inferred from the lambda's return type: here `Int`.
    print(callRet(lambda (x: Int) {} -> Int: x * 2, 6))
    # CHECK: 12

    # Inference also works for a capturing lambda.
    print(callRet(lambda (x: Int) {mut} -> Int: x + z, 7))
    # CHECK: 17

    # Capturing an enclosing function's parameter (rather than a local).
    addBase(100)
    # CHECK: 105

    # Nested lambda: the outer lambda's body invokes an inner lambda, which
    # captures the outer lambda's argument `x`.
    callInt(
        lambda (x: Int) {} -> Int: (lambda (y: Int) {imm x} -> Int: y + x)(3), 6
    )
    # CHECK: 9

    # Parametric lambda: the compile-time parameter `N` is bound at the call
    # site with `f[K]`.
    var pf = lambda [N: Int](x: Int) {} -> Int: x + N
    print(pf[5](3))
    # CHECK: 8

    # Nested lambda where the outer is parametric: the inner lambda references
    # the outer's compile-time parameter `N` directly.
    var pn = lambda [N: Int](x: Int) {} -> Int: (
        lambda (y: Int) {} -> Int: y + N
    )(x)
    print(pn[10](3))
    # CHECK: 13

    # A lambda referencing the compile-time parameter of an enclosing
    # parametric function.
    print(addEnclosingParam[7]())
    # CHECK: 10

    # Variadic arguments through the closure wrapper: `*args` positionally,
    # `**kwargs` as a keyword splat (packed into an OwnedKwargsDict).
    var va = lambda (*args: Int) {} -> Int: len(args)
    print(va(10, 20, 30))
    # CHECK: 3
    var kwl = lambda (var **kwargs: Int) {} -> Int: len(kwargs)
    print(kwl(a=1, b=2))
    # CHECK: 2
    var both = lambda (*args: Int, var **kwargs: Int) {} -> Int: len(
        args
    ) + len(kwargs)
    print(both(1, 2, a=3))
    # CHECK: 3
