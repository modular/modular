# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -verify-diagnostics %s


fn bind_fat_to_thin_target[g: fn (y: Int) -> Int](x: Int):
    pass


fn bind_fat_to_thin_main():
    var x = 4

    @__copy_capture(x)
    @parameter
    fn g(y: Int) -> Int:
        return x

    # expected-error @below {{cannot pass 'fn(y: Int) capturing -> Int' value, expected 'fn(y: Int) -> Int' in call parameter}}
    alias Bound = bind_fat_to_thin_target[g]
    Bound(3)


fn makeClosure(x: Int):
    var z = x+x

    @__copy_capture(z)
    @parameter
    fn writer() -> Int:
        # expected-error @below {{expression must be mutable in assignment}}
        z = z+z
        return z

    var y = writer()


@fieldwise_init
struct MemType:
    var a: Int

    fn foo(self) -> MemType:
        return MemType(self.a + self.a)


@register_passable
struct NoCopyType:
    var a: Int

    @implicit
    fn __init__(out self, aa: Int):
        self.a = aa

    fn foo(self) -> NoCopyType:
        return NoCopyType(self.a + self.a)


@no_inline
fn makeClosure(x: MemType):
    var rp: NoCopyType = NoCopyType(x.a)

    # expected-error @below {{value of type 'NoCopyType' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @below {{consider transferring the value with '^'}}
    @__copy_capture(rp)
    @parameter
    fn writer() -> Int:
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
