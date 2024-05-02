# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


fn bind_fat_to_thin_target[g: fn (y: int) -> int](x: int):
    pass


fn bind_fat_to_thin_main():
    var x = __mlir_attr.`4 : index`

    @__copy_capture(x)
    @parameter
    fn g(y: int) -> int:
        return x

    # expected-error @below {{cannot pass 'fn(y: index) capturing -> index' value, expected 'fn(y: index) -> index' in call parameter}}
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

    var y = formatter()


@value
struct MemType:
    var a: int

    fn foo(self) -> MemType:
        return MemType(__mlir_op.`index.add`(self.a, self.a))


@register_passable
struct NoCopyType:
    var a: int

    fn __init__(inout self, aa: int):
        self.a = aa

    fn foo(self) -> NoCopyType:
        return NoCopyType(__mlir_op.`index.add`(self.a, self.a))


@no_inline
fn makeClosure(x: MemType):
    var rp: NoCopyType = NoCopyType(x.a)

    # expected-error @below {{'NoCopyType' is not copyable because it has no '__copyinit__'}}
    @__copy_capture(rp)
    @parameter
    fn formatter() -> int:
        pass


fn bad_capture(x: Int):
    var z = x

    # expected-error @below {{cannot capture unknown value 'not_a_thing'}}
    @__copy_capture(not_a_thing)
    @parameter
    async fn closure_1():
        pass

    # expected-error @below {{cannot capture unknown value 'not_a_thing'}}
    @__move_capture(not_a_thing)
    @parameter
    async fn closure_2():
        pass
