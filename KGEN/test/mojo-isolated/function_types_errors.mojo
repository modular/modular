# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -split-input-file %s -verify-diagnostics


struct MemType:
    pass


fn mut_ship_function(mut x: MemType):
    ...


# We can convert from fn(read MemType)->None to fn(mut MemType)->None but not
# vice versa (see TTSMFS).
# expected-error @below {{cannot implicitly convert 'fn(mut x: MemType) -> None' value to 'fn(MemType) -> None' in alias initializer}}
alias read_ship_fn_alias: fn (read MemType) -> None = mut_ship_function


# // -----


# Tests that we detect too few arguments when doing function conversions.


# expected-note @below {{function declared here}}
fn infer_variadic[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyType, `>`], //,
    func: fn (x: Int, y: Int, * args: * ArgTypes) -> None,
]():
    pass


fn device_func(i: Int):
    pass


fn test_infer_variadic():
    # expected-error @below {{invalid call to 'infer_variadic': failed to infer parameter 'ArgTypes'}}
    # expected-note @below {{failed to infer parameter 'ArgTypes', parameter isn't used in any argument}}
    infer_variadic[device_func]()


# // -----


# Tests that we correctly match each incoming argument type against the
# callee's variadic's element trait.


# expected-note @below {{struct 'ZInt' does not implement all requirements for 'Sprongling'}}
struct ZInt:
    pass


# expected-note @below {{trait 'Sprongling' declared here}}
trait Sprongling:
    # expected-note @below {{required function 'sprongle' is not implemented}}
    fn sprongle(self):
        ...


# expected-note @below {{function declared here}}
fn infer_variadic[
    ArgTypes: __mlir_type[`!kgen.variadic<`, Sprongling, `>`], //,
    func: fn (* args: * ArgTypes) -> None,
]():
    pass


fn device_func(i: ZInt, j: ZInt):
    pass


# expected-error @below {{cannot bind type 'ZInt' to trait 'Sprongling'}}
fn test_infer_variadic():
    # expected-error @below {{invalid call to 'infer_variadic': failed to infer parameter 'ArgTypes'}}
    # expected-note @below {{failed to infer parameter 'ArgTypes', parameter isn't used in any argument}}
    infer_variadic[device_func]()


# // -----


fn device_func(a: Int, b: Bool) -> Int:
    return 73


@value
struct DeviceFunction[*ArgTypes: AnyRPTrivialType]:
    # expected-note @below {{function declared here}}
    fn call(self, *args: *ArgTypes) -> Int:
        return 91


fn compile[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyRPTrivialType, `>`], //,
    func: fn (* args: * ArgTypes) -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


fn main():
    var thing = compile[device_func]()
    # expected-error @below {{invalid call to 'call': method argument #1 cannot be converted from 'StringLiteral["hello"]' to 'Bool'}}
    var result2 = thing.call(42, "hello")


# // -----

# Tests that we reject any device_function that is still generic.


# A function that has a T that can't be inferred from anything
fn device_func_unusedT[T: AnyType](a: Int, b: Bool) -> Int:
    return 73


# A function that has a T that can be inferred from arguments
fn device_func_usedT[T: AnyType](a: T, b: Bool) -> Int:
    return 73


@value
struct DeviceFunction[*ArgTypes: AnyRPTrivialType]:
    pass


# expected-note @below {{function declared here}}
fn compile[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyRPTrivialType, `>`], //,
    func: fn (* args: * ArgTypes) -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


fn test_reject_generic_device_func_unusedT():
    # TODO(MOCO-1828): Better error message.
    # expected-error @below {{invalid call to 'compile': failed to infer parameter 'ArgTypes'}}
    # expected-note @below {{failed to infer parameter 'ArgTypes', parameter isn't used in any argument}}
    var thing = compile[device_func_unusedT]()


# Slightly different case, for no particular reason
fn test_reject_generic_device_func_usedT():
    # TODO(MOCO-1828): Better error message.
    # expected-error @below {{invalid call to 'compile': failed to infer parameter 'ArgTypes'}}
    # expected-note @below {{failed to infer parameter 'ArgTypes', parameter isn't used in any argument}}
    var thing = compile[device_func_usedT]()


# // -----


fn device_func(a: Int, b: Bool) -> Int:
    return 73


@value
struct DeviceFunction[*ArgTypes: AnyRPTrivialType]:
    # expected-note @below {{function declared here}}
    fn call(self, *args: *ArgTypes) -> Int:
        return 91


fn compile[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyRPTrivialType, `>`], //,
    func: fn (* args: * ArgTypes) -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


fn main():
    var thing = compile[device_func]()
    # expected-error @below {{invalid call to 'call': method argument #1 cannot be converted from 'StringLiteral["hello"]' to 'Bool'}}
    var result2 = thing.call(42, "hello")


# // -----


# Tests a GPU-function-like case (see FTAGPUF) but this one catches when the
# user hands in something of the wrong type.
# TODO(MOCO-1106): This doesn't catch when the given argument type
# (DeviceBuffer[Int]) mismatches the expected argument type
# (UnsafePointer[Float32]) by only the input parameter (Int vs Float32), that's
# only caught by elaborator for now and is tested elsewhere (search FTAGPUF).


@value
struct ZBool:
    pass


# Copied from stdlib
@always_inline("nodebug")
fn rebind[
    src_type: AnyTrivialRegType, //,
    dest_type: AnyTrivialRegType,
](src: src_type) -> dest_type:
    return __mlir_op.`kgen.rebind`[_type=dest_type](src)


trait ConvertibleToZPointer:
    alias Pointee: AnyType

    fn to_zpointer(self) -> ZPointer[Pointee]:
        pass


@register_passable("trivial")
struct ZPointer[T: AnyType](AnyRPTrivialType):
    fn __init__(out self):
        pass

    @implicit
    fn __init__[C: ConvertibleToZPointer](out self, c: C):
        # TODO(MOCO-1106): If we can remove this rebind, we win. We'd need to
        # constrain C.Pointee=T somehow, or make ConvertibleToZPointer into a
        # generic trait instead of using an associated alias.
        # As it is, this won't catch incorrectly passing in a e.g.
        # ZDeviceBuffer[Int] into a ZPointer[Bool].
        var z: ZPointer[T] = rebind[ZPointer[T]](c.to_zpointer())


trait ConvertibleToZLayoutTensor:
    fn to_tensor(self) -> ZLayoutTensor:
        pass


@register_passable("trivial")
struct ZLayoutTensor(AnyRPTrivialType):
    fn __init__(out self):
        pass

    @implicit
    fn __init__[C: ConvertibleToZLayoutTensor](out self, c: C):
        var z: ZLayoutTensor = c.to_tensor()


@value
struct DeviceFunction[*ArgTypes: AnyRPTrivialType]:
    # expected-note @below {{function declared here}}
    fn call(self, *args: *ArgTypes) -> Int:
        return 91


@value
struct ManagedLayoutTensor(ConvertibleToZLayoutTensor):
    fn to_tensor(self) -> ZLayoutTensor:
        return ZLayoutTensor()


# Never converted, the GPU just uses this one directly
@value
@register_passable("trivial")
struct NDBuffer(AnyRPTrivialType):
    pass


fn kernel(t: ZLayoutTensor, p: ZPointer[Int], n: NDBuffer) -> Int:
    return 73


fn compile[
    ArgTypes: __mlir_type[`!kgen.variadic<`, AnyRPTrivialType, `>`], //,
    func: fn (* args: * ArgTypes) -> Int,
]() -> DeviceFunction[*ArgTypes]:
    return DeviceFunction[*ArgTypes]()


fn main():
    var thing = compile[kernel]()
    var mlt = ManagedLayoutTensor()
    var ndb = NDBuffer()
    # This ZBool() is incorrect, not even close to the ZPointer[Int] that's
    # expected.
    # expected-error @below {{invalid call to 'call': method argument #1 cannot be converted from 'ZBool' to 'ZPointer[Int]'}}
    var result1 = thing.call(mlt, ZBool(), ndb)
