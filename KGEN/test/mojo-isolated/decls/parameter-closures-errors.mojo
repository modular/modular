# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %translate-with-packages -verify-diagnostics %s


fn bind_fat_to_thin_target[g: fn (y: int) -> int](x: int):
    pass


fn bind_fat_to_thin_main():
    let x = __mlir_attr.`4 : index`

    @__copy_capture(x)
    @parameter
    fn g(y: int) -> int:
        return x

    # expected-error @below {{cannot pass 'fn(y = index) capturing -> index' value, parameter expected 'fn(y = index) -> index'}}
    alias Bound = bind_fat_to_thin_target[g]
    Bound(3)


fn makeClosure(x: int):
    var z = __mlir_op.`index.add`(x, x)

    @__copy_capture(z)
    @parameter
    fn formatter() -> int:
        # expected-error @below {{expression must be mutable in assignment}}
        z = __mlir_op.`index.add`(z, z)
        return z

    let y = formatter()


@value
struct MemType:
    var a: int

    fn foo(self) -> MemType:
        return MemType(__mlir_op.`index.add`(self.a, self.a))


@register_passable
struct NoCopyType:
    var a: int

    fn __init__(aa: int) -> Self:
        return NoCopyType {a: aa}

    fn foo(self) -> NoCopyType:
        return NoCopyType(__mlir_op.`index.add`(self.a, self.a))


@no_inline
fn makeClosure(x: MemType):
    var z: MemType = x.foo()
    var rp: NoCopyType = NoCopyType(x.a)

    # expected-error @below {{cannot capture 'z' because capturing instances of memory only types in parametric functions is not supported}}
    # expected-error @below {{cannot capture 'rp'}}
    # expected-error @below {{'NoCopyType' does not implement the '__copyinit__' method}}
    @__copy_capture(z, rp)
    @parameter
    fn formatter() -> int:
        return z.a

    let y = formatter()


fn makeClosureWithCaptureLetWarn(x: int):
    let z = x

    @parameter
    async fn formatter() -> int:
        # expected-error @below {{cannot capture let without copy: z}}
        return z
