# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# An end-to-end test case mocking a kernel invocation via a wrapper function
# that converts a C void** argument into a Mojo VariadicPack. The main feature
# being tested is that `wrapped_entry_point` is expressible.
#
# The wrapper is parameterized by the kernel signature so that the expected
# types can be extracted, and acted upon differently depending on how the driver
# encodes these arguments in the void** argument.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s

from std.collections import InlineArray
from std.reflection import get_base_type_name


@fieldwise_init
struct KernelFunction[
    declared_arg_types: Variadic.TypesOfTrait[AnyType],
    declared_ret_type: RegisterPassable,
    //,
    func: def(
        * args: * TypeList[declared_arg_types]()
    ) thin -> declared_ret_type,
]:
    pass


# A wrapped void** argument.
struct KernelArgPack[kernel: KernelFunction[_]]:
    var pointers: InlineArray[
        MutOpaquePointer[MutExternalOrigin],
        TypeList[Self.kernel.declared_arg_types].size,
    ]

    def __init__(out self):
        self.pointers = type_of(self.pointers)(
            fill=MutOpaquePointer[MutExternalOrigin]()
        )


# Rudimentary check to see if a type is an UnsafePointer.
def looks_like_pointer[T: AnyType]() -> Bool:
    comptime base_name = get_base_type_name[T]()
    return base_name == "UnsafePointer"


# Wrapped kernel entry point parameterized by the actual kernel to be invoked.
# Takes in a void** argument and converts it into a Mojo VariadicPack and calls
# the actual kernel function.
def wrapped_entry_point[
    kernel: KernelFunction[_]
](
    pa: UnsafePointer[KernelArgPack[kernel], MutAnyOrigin],
) -> kernel.declared_ret_type:
    comptime to_unsafe_pointer_mapper[
        T: AnyType
    ]: Movable & Defaultable & ImplicitlyCopyable = UnsafePointer[
        T, MutExternalOrigin
    ]
    comptime UnsafePointerTupleType = Tuple[
        *Variadic.map_types_to_types[
            kernel.declared_arg_types, to_unsafe_pointer_mapper
        ]().upcast[Movable]()
    ]
    var ptr_tuple: UnsafePointerTupleType = {}

    comptime for i in range(TypeList[kernel.declared_arg_types].size):
        comptime ArgType = kernel.declared_arg_types[i]
        comptime if looks_like_pointer[ArgType]():
            ptr_tuple[i] = rebind[type_of(ptr_tuple[i])](
                pa[].pointers.unsafe_ptr() + i
            )
        else:
            ptr_tuple[i] = rebind[type_of(ptr_tuple[i])](pa[].pointers[i])

    comptime PackType = VariadicPack[
        origin=MutExternalOrigin,
        element_trait=AnyType,
        False,
        *TypeList[kernel.declared_arg_types](),
    ]
    var raw_pack = __mlir_op.`lit.ref.pack.from_pointer_pack`[
        _type=PackType._mlir_type
    ](ptr_tuple._mlir_value)
    var pack = PackType(raw_pack)
    return kernel.func(*pack)


# Mimic invoking the wrapped kernel from FFI via a void** argument.
def invoke_kernel[
    kernel: KernelFunction[_]
](
    wrapped_kernel: def(
        UnsafePointer[KernelArgPack[kernel], MutAnyOrigin],
    ) thin -> kernel.declared_ret_type,
    *args: *TypeList[kernel.declared_arg_types](),
) -> kernel.declared_ret_type:
    var pa = KernelArgPack[kernel]()
    comptime for i in range(TypeList[kernel.declared_arg_types].size):
        comptime ArgType = kernel.declared_arg_types[i]
        comptime if looks_like_pointer[ArgType]():
            pa.pointers[i] = rebind[MutOpaquePointer[MutExternalOrigin]](
                args[i]
            )
        else:
            pa.pointers[i] = rebind[MutOpaquePointer[MutExternalOrigin]](
                UnsafePointer(to=args[i])
            )
    return wrapped_kernel(
        UnsafePointer(to=pa),
    )


comptime IntPtr = UnsafePointer[Int, MutAnyOrigin]
comptime ScalarKernel = KernelFunction[scalar_kernel]()
comptime MixedKernel = KernelFunction[mixed_kernel]()


def scalar_kernel(x: Int) -> Int:
    return x + 1


def mixed_kernel(x: Int, p: IntPtr) -> Int:
    return x + p[]


def main():
    comptime WrappedScalarKernel = wrapped_entry_point[ScalarKernel]
    # CHECK: 42
    print(invoke_kernel[ScalarKernel](WrappedScalarKernel, 41))

    var value = 7
    comptime WrappedMixedKernel = wrapped_entry_point[MixedKernel]
    # CHECK: 11
    print(
        invoke_kernel[MixedKernel](
            WrappedMixedKernel, 4, UnsafePointer(to=value)
        )
    )
