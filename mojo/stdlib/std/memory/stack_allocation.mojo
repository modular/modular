# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Defines the `stack_allocation` function for stack-based memory allocation.

You can import these APIs from the `memory` package. For example:

```mojo
from std.memory import stack_allocation
```
"""

from std.collections.string.string_slice import _get_kgen_string
from std.sys import align_of, is_gpu


# TODO(MSTDL-2015): ASAN error when updating to use `UnsafePointer`.
@always_inline
def stack_allocation[
    count: Int,
    dtype: DType,
    /,
    alignment: Int = align_of[dtype](),
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[
    Scalar[dtype],
    MutExternalOrigin,
    address_space=address_space,
]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        dtype: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return stack_allocation[
        count, Scalar[dtype], alignment=alignment, address_space=address_space
    ]()


# TODO(MSTDL-2015): ASAN error when updating to use `UnsafePointer`.
@always_inline
def stack_allocation[
    count: Int,
    type: AnyType,
    /,
    name: Optional[StaticString] = None,
    alignment: Int = align_of[type](),
    address_space: AddressSpace = AddressSpace.GENERIC,
]() -> UnsafePointer[type, MutExternalOrigin, address_space=address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        name: The name of the global variable (only honored in certain cases).
        alignment: Address alignment of the allocated data.
        address_space: The address space of the pointer.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    comptime if is_gpu():
        # On NVGPU, SHARED and CONSTANT address spaces lower to global memory.

        comptime global_name = name.value() if name else "_global_alloc"

        comptime if address_space == AddressSpace.SHARED:
            return __mlir_op.`pop.global_alloc`[
                name=_get_kgen_string[global_name](),
                count=count._mlir_value,
                memoryType=__mlir_attr.`#pop<global_alloc_addr_space gpu_shared>`,
                _type=UnsafePointer[
                    type, MutExternalOrigin, address_space=address_space
                ]._mlir_type,
                alignment=alignment._mlir_value,
            ]()
        elif address_space == AddressSpace.CONSTANT:
            # No need to annotation this global_alloc because constants in
            # GPU shared memory won't prevent llvm module splitting to
            # happen since they are immutables.
            return __mlir_op.`pop.global_alloc`[
                name=_get_kgen_string[global_name](),
                count=count._mlir_value,
                _type=UnsafePointer[
                    type, MutExternalOrigin, address_space=address_space
                ]._mlir_type,
                alignment=alignment._mlir_value,
            ]()

        # MSTDL-797: The NVPTX backend requires that `alloca` instructions may
        # only have generic address spaces. When allocating LOCAL memory,
        # addrspacecast the resulting pointer.
        elif address_space == AddressSpace.LOCAL:
            var generic_ptr = __mlir_op.`pop.stack_allocation`[
                count=count._mlir_value,
                _type=UnsafePointer[type, MutExternalOrigin]._mlir_type,
                alignment=alignment._mlir_value,
            ]()
            return __mlir_op.`pop.pointer.bitcast`[
                _type=UnsafePointer[
                    type, MutExternalOrigin, address_space=address_space
                ]._mlir_type
            ](generic_ptr)

    # Perform a stack allocation of the requested size, alignment, and type.
    return __mlir_op.`pop.stack_allocation`[
        count=count._mlir_value,
        _type=UnsafePointer[
            type, MutExternalOrigin, address_space=address_space
        ]._mlir_type,
        alignment=alignment._mlir_value,
    ]()
