# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs."""

from Assert import assert_param
from DType import DType
from Intrinsics import llvm_intrinsic
from Pointer import DTypePointer
from SIMD import SIMD, Int32
from Math import is_power_of_2
from Memory import stack_allocation as _generic_stack_allocation
from TargetInfo import simdwidthof, alignof, sizeof


# ===----------------------------------------------------------------------===#
# Address Space
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct AddressSpace:
    var _value: Int

    # See https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space
    alias GENERIC = AddressSpace(0)
    """Generic address space."""
    alias GLOBAL = AddressSpace(1)
    """Global address space."""
    alias CONSTANT = AddressSpace(2)
    """Constant address space."""
    alias SHARED = AddressSpace(3)
    """Shared address space."""
    alias PARAM = AddressSpace(4)
    """Param address space."""
    alias LOCAL = AddressSpace(5)
    """Local address space."""

    fn __init__(value: Int) -> Self:
        return Self {_value: value}

    fn value(self) -> Int:
        """The integral value of the address space.

        Returns:
          The integral value of the address space.
        """
        return self._value

    fn __eq__(self, other: AddressSpace) -> Bool:
        """The True if the two address spaces are equal and False otherwise.

        Returns:
          True if the two address spaces are equal and False otherwise.
        """
        return self.value() == other.value()


# ===----------------------------------------------------------------------===#
# stack allocation
# ===----------------------------------------------------------------------===#


@always_inline
fn stack_allocation[
    count: Int, type: DType, alignment: Int, address_space: AddressSpace
]() -> DTypeDevicePointer[type, address_space]:
    return _stack_allocation[count, type, alignment, address_space]()


@always_inline
fn stack_allocation[
    count: Int, type: DType, address_space: AddressSpace
]() -> DTypeDevicePointer[type, address_space]:
    return _stack_allocation[count, type, 1, address_space]()


@always_inline
fn stack_allocation[count: Int, type: DType]() -> DTypePointer[type]:
    return _stack_allocation[count, type, 1]()


# ===----------------------------------------------------------------------===#
# ThreadIdx
# ===----------------------------------------------------------------------===#


struct ThreadIdx:
    """ThreadIdx provides static methods for getting the x/y/z coordinates of
    a thread within a block."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` coordinate of the thread within the block.

        Returns: The `x` coordinate within the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.tid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` coordinate of the thread within the block.

        Returns: The `y` coordinate within the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.tid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` coordinate of the thread within the block.

        Returns: The `z` coordinate within the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.tid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# BlockIdx
# ===----------------------------------------------------------------------===#


struct BlockIdx:
    """BlockIdx provides static methods for getting the x/y/z coordinates of
    a block within a grid."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` coordinate of the block within a grid.

        Returns: The `x` coordinate within the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ctaid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` coordinate of the block within a grid.

        Returns: The `y` coordinate within the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ctaid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` coordinate of the block within a grid.

        Returns: The `z` coordinate within the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ctaid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# BlockDim
# ===----------------------------------------------------------------------===#


struct BlockDim:
    """BlockDim provides static methods for getting the x/y/z dimension of a
    block."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` dimension of the block.

        Returns: The `x` dimension of the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ntid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` dimension of the block.

        Returns: The `y` dimension of the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ntid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` dimension of the block.

        Returns: The `z` dimension of the block.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.ntid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# GridDim
# ===----------------------------------------------------------------------===#


struct GridDim:
    """GridDim provides static methods for getting the x/y/z dimension of a
    grid."""

    @staticmethod
    @always_inline("nodebug")
    fn x() -> Int:
        """Gets the `x` dimension of the grid.

        Returns: The `x` dimension of the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.nctaid.x", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn y() -> Int:
        """Gets the `y` dimension of the grid.

        Returns: The `y` dimension of the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.nctaid.y", Int32]().value

    @staticmethod
    @always_inline("nodebug")
    fn z() -> Int:
        """Gets the `z` dimension of the grid.

        Returns: The `z` dimension of the grid.
        """
        return llvm_intrinsic["llvm.nvvm.read.ptx.sreg.nctaid.z", Int32]().value


# ===----------------------------------------------------------------------===#
# barrier
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn barrier():
    """Performs a synchronization barrier on block (equivelent to `__syncthreads`
    in CUDA).
    """
    llvm_intrinsic["llvm.nvvm.barrier0", NoneType]()


# ===----------------------------------------------------------------------===#
# CONTENT AFTER THOS NEEDS TO BE DELETED
# ===----------------------------------------------------------------------===#
# TODO: Delete me, but to do that you need a combination of things. 1: we need
# to have keyword parameters so that we can write Pointer[type, address_space=0]
# that would allow us to unify the stack allocations that take an address space
# (defined here) with the ones that do not take an address space (defined in
# Memory.mojo).


# ===----------------------------------------------------------------------===#
# stack_allocation
# ===----------------------------------------------------------------------===#


@always_inline
fn _stack_allocation[
    count: Int,
    type: DType,
    alignment: Int,
]() -> DTypePointer[type]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        alignment: Address alignment of the allocated data.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return _generic_stack_allocation[count, type, alignment]()


@always_inline
fn _stack_allocation[
    count: Int,
    type: DType,
    alignment: Int,
    address_space: AddressSpace,
]() -> DTypeDevicePointer[type, address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the allocated data.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    return _stack_allocation[
        count,
        __mlir_type[`!pop.scalar<`, type.value, `>`],
        alignment,
        address_space,
    ]().address


@always_inline
fn _stack_allocation[
    count: Int, type: AnyType, alignment: Int, address_space: AddressSpace
]() -> DevicePointer[type, address_space]:
    """Allocates data buffer space on the stack given a data type and number of
    elements.

    Parameters:
        count: Number of elements to allocate memory for.
        type: The data type of each element.
        alignment: Address alignment of the allocated data.
        address_space: The address space of the allocated data.

    Returns:
        A data pointer of the given type pointing to the allocated space.
    """

    @parameter
    if address_space == AddressSpace.SHARED:
        return __mlir_op.`pop.global_alloc`[
            count : count.value,
            _type : __mlir_type[
                `!pop.pointer<`, type, `,`, address_space.value().value, `>`
            ],
            alignment : alignment.value,
            address_space : address_space.value().value,
        ]()
    else:
        return __mlir_op.`pop.stack_allocation`[
            count : count.value,
            _type : __mlir_type[
                `!pop.pointer<`, type, `,`, address_space.value().value, `>`
            ],
            alignment : alignment.value,
            address_space : address_space.value().value,
        ]()


# ===----------------------------------------------------------------------===#
# Pointer
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct DevicePointer[type: AnyType, address_space: AddressSpace]:
    """Defines a Pointer struct that contains an address of any mlirtype at the
    specified address space.

    Parameters:
        type: Type of the underlying data.
        address_space: The address space of the pointer.
    """

    alias pointer_type = __mlir_type[
        `!pop.pointer<`, type, `,`, address_space.value().value, `>`
    ]

    var address: Self.pointer_type
    """The pointed-to address."""

    @always_inline
    fn __init__() -> Self:
        """Constructs a null Pointer from the value of pop.pointer type.

        Returns:
            Constructed Pointer object.
        """
        return Self.get_null()

    @always_inline
    fn __init__(address: Self.pointer_type) -> Self:
        """Constructs a Pointer from the address.

        Args:
            address: The input pointer address.

        Returns:
            Constructed Pointer object.
        """
        return Self {address: address}

    @always_inline
    fn __init__(value: SIMD[DType.address, 1]) -> Self:
        """Constructs a Pointer from the value of scalar address.

        Args:
            value: The input pointer index.

        Returns:
            Constructed Pointer object.
        """
        let address = __mlir_op.`pop.index_to_pointer`[
            _type : Self.pointer_type
        ](value.cast[DType.index]().value)
        return Self {address: address}

    @staticmethod
    @always_inline
    fn get_null() -> Self:
        """Constructs a Pointer representing nullptr.

        Returns:
            Constructed nullptr Pointer object.
        """
        return __mlir_attr[`#M.pointer<0> : `, Self.pointer_type]

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the pointer is null.

        Returns:
            Returns False if the pointer is null and True otherwise.
        """
        return self != Self.get_null()

    @staticmethod
    @always_inline
    fn address_of(inout arg: type) -> Self:
        """Gets the address of the argument.

        Args:
            arg: The value to get the address of.

        Returns:
            A pointer struct which contains the address of the argument.
        """
        return __mlir_op.`pop.pointer.bitcast`[_type : Self.pointer_type](
            __get_lvalue_as_address(arg)
        )

    @always_inline
    fn __getitem__(self, offset: Int) -> type:
        """Loads the value the Pointer object points to with the given offset.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.load(offset)

    @always_inline
    fn load(self, offset: Int) -> type:
        """Loads the value the Pointer object points to with the given offset.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.offset(offset).load()

    @always_inline("nodebug")
    fn load(self) -> type:
        """Loads the value the Pointer object points to.

        Returns:
            The loaded value.
        """
        return __mlir_op.`pop.load`(self.address)

    @always_inline
    fn store(self, offset: Int, value: type):
        """Stores the specified value to the location the Pointer object points
        to with the given offset.

        Args:
            offset: The offset to load from.
            value: The value to store.
        """
        self.offset(offset).store(value)
        return

    @always_inline("nodebug")
    fn store(self, value: type):
        """Stores the specified value to the location the Pointer object points
        to.

        Args:
            value: The value to store.
        """
        __mlir_op.`pop.store`(value, self.address)
        return

    @always_inline
    fn __as_index(self) -> Int:
        # Returns the pointer address as an index.
        let addr = llvm_intrinsic[
            "addrspacecast", __mlir_type[`!pop.pointer<`, type, `>`]
        ](self.address)
        return __mlir_op.`pop.pointer_to_index`[
            _type : __mlir_type.`!pop.scalar<index>`
        ](addr)

    # ===------------------------------------------------------------------=== #
    # Casting
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn bitcast[
        new_type: AnyType
    ](self) -> DevicePointer[new_type, address_space]:
        """Bitcasts a Pointer to a different type.

        Parameters:
            new_type: The target type.

        Returns:
            A new Pointer object with the specified type and the same address,
            as the original Pointer.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type : __mlir_type[
                `!pop.pointer<`, new_type, `,`, address_space.value().value, `>`
            ]
        ](self.address)

    # ===------------------------------------------------------------------=== #
    # Comparisons
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.__as_index() == rhs.__as_index()

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.

        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.__as_index() != rhs.__as_index()

    # ===------------------------------------------------------------------=== #
    # Pointer Arithmetic
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn offset(self, idx: Int) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Args:
            idx: The offset.

        Returns:
            The new Pointer shifted by the offset.
        """
        # Returns a new pointer shifted by the specified offset.
        return __mlir_op.`pop.offset`(self.address, idx.value)

    @always_inline
    fn __add__(self, rhs: Int) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Args:
            rhs: The offset.

        Returns:
            The new Pointer shifted by the offset.
        """
        return self.offset(rhs)

    @always_inline
    fn __sub__(self, rhs: Int) -> Self:
        """Returns a new pointer shifted back by the specified offset.

        Args:
            rhs: The offset.

        Returns:
            The new Pointer shifted back by the offset.
        """
        return self.offset(-rhs)

    @always_inline
    fn __iadd__(inout self, rhs: Int):
        """Shifts the current pointer by the specified offset.

        Args:
            rhs: The offset.
        """
        self = self + rhs

    @always_inline
    fn __isub__(inout self, rhs: Int):
        """Shifts back the current pointer by the specified offset.

        Args:
            rhs: The offset.
        """
        self = self - rhs


# ===----------------------------------------------------------------------===#
# DTypeDevicePointer
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct DTypeDevicePointer[type: DType, address_space: AddressSpace]:
    """Defines a `DTypeDevicePointer` struct that contains an address of the
    given dtype at the specified address space.

    Parameters:
        type: DType of the underlying data.
        address_space: The address space of the pointer.
    """

    alias element_type = __mlir_type[`!pop.scalar<`, type.value, `>`]
    alias pointer_type = __mlir_type[
        `!pop.pointer<`,
        Self.element_type,
        `,`,
        address_space.value().value,
        `>`,
    ]
    var address: Self.pointer_type
    """The pointed-to address."""

    @always_inline("nodebug")
    fn __init__() -> Self:
        """Constructs a null `DTypePointer` from the given type.

        Returns:
            Constructed `DTypePointer` object.
        """

        return Self.get_null()

    @always_inline("nodebug")
    fn __init__(address: Self.pointer_type) -> Self:
        """Constructs a `DTypePointer` from the given address.

        Args:
            address: The input pointer.

        Returns:
            Constructed `DTypePointer` object.
        """
        # Construct a pointer type.
        return Self {address: address}

    @always_inline("nodebug")
    fn __init__(value: DevicePointer[SIMD[type, 1], address_space]) -> Self:
        """Constructs a `DTypePointer` from a scalar pointer of the same type.

        Args:
            value: The scalar pointer.

        Returns:
            Constructed `DTypePointer`.
        """
        return value.bitcast[
            __mlir_type[`!pop.scalar<`, type.value, `>`]
        ]().address

    @always_inline
    fn __init__(value: SIMD[DType.address, 1]) -> Self:
        """Constructs a `DTypePointer` from the value of scalar address.

        Args:
            value: The input pointer index.

        Returns:
            Constructed `DTypePointer` object.
        """
        let address = __mlir_op.`pop.index_to_pointer`[
            _type : Self.pointer_type
        ](value.cast[DType.index]().value)
        return Self {address: address}

    @staticmethod
    @always_inline
    fn get_null() -> Self:
        """Constructs a `DTypePointer` representing *nullptr*.

        Returns:
            Constructed *nullptr* `DTypePointer` object.
        """
        return __mlir_attr[`#M.pointer<0> : `, Self.pointer_type]

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the pointer is *null*.

        Returns:
            Returns False if the pointer is *null* and True otherwise.
        """
        return self != Self.get_null()

    @staticmethod
    @always_inline
    fn address_of(
        inout arg: __mlir_type[`!pop.scalar<`, type.value, `>`]
    ) -> Self:
        """Gets the address of the argument.

        Args:
            arg: The value to get the address of.

        Returns:
            A pointer struct which contains the address of the argument.
        """
        return __mlir_op.`pop.pointer.bitcast`[_type : Self.pointer_type](
            __get_lvalue_as_address(arg)
        )

    # ===------------------------------------------------------------------=== #
    # Comparisons
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are equal.

        Args:
            rhs: The value of the other pointer.


        Returns:
            True if the two pointers are equal and False otherwise.
        """
        return self.__as_index() == rhs.__as_index()

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
        """Returns True if the two pointers are not equal.

        Args:
            rhs: The value of the other pointer.


        Returns:
            True if the two pointers are not equal and False otherwise.
        """
        return self.__as_index() != rhs.__as_index()

    @always_inline
    fn __lt__(self, rhs: Self) -> Bool:
        """Returns True if this pointer represents a lower address than rhs.

        Args:
            rhs: The value of the other pointer.


        Returns:
            True if this pointer represents a lower address and False otherwise.
        """
        return self.__as_index() < rhs.__as_index()

    # ===------------------------------------------------------------------=== #
    # Casting
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn bitcast[new_type: DType](self) -> DTypePointer[new_type]:
        """Bitcasts `DTypePointer` to a different dtype.

        Parameters:
            new_type: The target dtype.

        Returns:
            A new `DTypePointer` object with the specified dtype and the same
            address, as the original `DTypePointer`.
        """
        return __mlir_op.`pop.pointer.bitcast`[
            _type : __mlir_type[`!pop.pointer<scalar<`, new_type.value, `>>`]
        ](self.address)

    @always_inline
    fn as_scalar_pointer(self) -> DevicePointer[SIMD[type, 1], address_space]:
        """Converts the `DTypePointer` to a scalar pointer of the same dtype.

        Returns:
            A `Pointer` to a scalar of the same dtype.
        """
        return DevicePointer[
            __mlir_type[`!pop.scalar<`, type.value, `>`], address_space
        ](self.address).bitcast[SIMD[type, 1]]()

    # ===------------------------------------------------------------------=== #
    # Load/Store
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn load(self, offset: Int) -> SIMD[type, 1]:
        """Loads a single element (SIMD of size 1) from the pointer at the
        specified index.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.offset(offset).load()

    @always_inline
    fn load(self) -> SIMD[type, 1]:
        """Loads a single element (SIMD of size 1) from the pointer.

        Returns:
            The loaded value.
        """

        return self.simd_load[1]()

    @always_inline
    fn simd_load[width: Int](self, offset: Int) -> SIMD[type, width]:
        """Loads a SIMD vector of elements from the pointer at the specified
        offset.

        Parameters:
            width: The SIMD width.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded value.
        """
        return self.offset(offset).simd_load[width]()

    @always_inline
    fn simd_load[width: Int](self) -> SIMD[type, width]:
        """Loads a SIMD vector of elements from the pointer.

        Parameters:
            width: The SIMD width.

        Returns:
            The loaded SIMD value.
        """
        return self.aligned_simd_load[width, 1]()

    @always_inline
    fn aligned_simd_load[
        width: Int, alignment: Int
    ](self, offset: Int) -> SIMD[type, width]:
        """Loads a SIMD vector of elements from the pointer at the specified
        offset with the guaranteed specified alignment.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The offset to load from.

        Returns:
            The loaded SIMD value.
        """
        return self.offset(offset).aligned_simd_load[width, alignment]()

    @always_inline
    fn aligned_simd_load[width: Int, alignment: Int](self) -> SIMD[type, width]:
        """Loads a SIMD vector of elements from the pointer with the guaranteed
        specified alignment.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Returns:
            The loaded SIMD value.
        """
        # Loads a simd value from the pointer.
        alias alignment_value = alignment.value
        let ptr = __mlir_op.`pop.pointer.bitcast`[
            _type : __mlir_type[
                `!pop.pointer<`,
                SIMD[type, width],
                `,`,
                address_space.value().value,
                `>`,
            ]
        ](self.address)
        let result = __mlir_op.`pop.load`[alignment:alignment_value](ptr)
        return result

    @always_inline
    fn store(self, offset: Int, val: SIMD[type, 1]):
        """Stores a single element value at the given offset.

        Args:
            offset: The offset to store to.
            val: The value to store.
        """
        self.offset(offset).store(val)

    @always_inline
    fn store(self, val: SIMD[type, 1]):
        """Stores a single element value.

        Args:
            val: The value to store.
        """
        self.simd_store[1](val)

    @always_inline
    fn simd_store[width: Int](self, offset: Int, val: SIMD[type, width]):
        """Stores a SIMD vector at the given offset.

        Parameters:
            width: The SIMD width.

        Args:
            offset: The offset to store to.
            val: The SIMD value to store.
        """
        self.offset(offset).simd_store[width](val)

    @always_inline
    fn simd_store[width: Int](self, val: SIMD[type, width]):
        """Stores a SIMD vector.

        Parameters:
            width: The SIMD width.

        Args:
            val: The SIMD value to store.
        """
        self.aligned_simd_store[width, 1](val)

    @always_inline
    fn aligned_simd_store[
        width: Int, alignment: Int
    ](self, offset: Int, val: SIMD[type, width]):
        """Stores a SIMD vector at the given offset with a guaranteed alignment.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            offset: The offset to store to.
            val: The SIMD value to store.
        """
        self.offset(offset).aligned_simd_store[width, alignment](val)

    @always_inline
    fn aligned_simd_store[
        width: Int, alignment: Int
    ](self, val: SIMD[type, width]):
        """Stores a SIMD vector with a guaranteed alignment.

        Parameters:
            width: The SIMD width.
            alignment: The minimal alignment of the address.

        Args:
            val: The SIMD value to store.
        """
        alias alignment_value = alignment.value
        let ptr = __mlir_op.`pop.pointer.bitcast`[
            _type : __mlir_type[
                `!pop.pointer<`,
                SIMD[type, width],
                `,`,
                address_space.value().value,
                `>`,
            ]
        ](self.address)
        __mlir_op.`pop.store`[alignment:alignment_value](val, ptr)

    @always_inline
    fn __as_index(self) -> Int:
        # Returns the pointer address as an index.
        let addr = llvm_intrinsic[
            "addrspacecast", __mlir_type[`!pop.pointer<`, SIMD[type, 1], `>`]
        ](self.address)
        return __mlir_op.`pop.pointer_to_index`[
            _type : __mlir_type.`!pop.scalar<index>`
        ](addr)

    @always_inline
    fn is_aligned[alignment: Int](self) -> Bool:
        """Checks if the pointer is aligned.

        Parameters:
            alignment: The minimal desired alignment.

        Returns:
            `True` if the pointer is at least `alignment`-aligned or `False`
            otherwise.
        """
        assert_param[
            is_power_of_2(alignment), "alignment must be a power of 2."
        ]()
        return self.__as_index() % alignment == 0

    # ===------------------------------------------------------------------=== #
    # Pointer Arithmetic
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn offset(self, idx: Int) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Args:
            idx: The offset of the new pointer.

        Returns:
            The new constructed DTypePointer.
        """
        return __mlir_op.`pop.offset`(self.address, idx.value)

    @always_inline
    fn __add__(self, rhs: Int) -> Self:
        """Returns a new pointer shifted by the specified offset.

        Args:
            rhs: The offset.

        Returns:
            The new DTypePointer shifted by the offset.
        """
        return self.offset(rhs)

    @always_inline
    fn __sub__(self, rhs: Int) -> Self:
        """Returns a new pointer shifted back by the specified offset.

        Args:
            rhs: The offset.

        Returns:
            The new DTypePointer shifted by the offset.
        """
        return self.offset(-rhs)

    @always_inline
    fn __iadd__(inout self, rhs: Int):
        """Shifts the current pointer by the specified offset.

        Args:
            rhs: The offset.
        """
        self = self + rhs

    @always_inline
    fn __isub__(inout self, rhs: Int):
        """Shifts back the current pointer by the specified offset.

        Args:
            rhs: The offset.
        """
        self = self - rhs
