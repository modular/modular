# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from collections import OptionalReg
from sys import alignof, bitwidthof, sizeof, triple_is_nvidia_cuda
from sys._assembly import inlined_assembly
from sys.intrinsics import _RegisterPackType

from builtin.dtype import _uint_type_of_width
from memory import UnsafePointer
from memory.pointer import AddressSpace as _AddressSpace
from memory.pointer import _GPUAddressSpace
from memory.unsafe import bitcast

from utils import IndexList, StaticTuple

from ._utils import to_llvm_ptr, to_llvm_shared_mem_ptr, to_i32

# ===----------------------------------------------------------------------===#
# AddressSpace
# ===----------------------------------------------------------------------===#

alias AddressSpace = _GPUAddressSpace

# ===----------------------------------------------------------------------===#
# cp.async
# ===----------------------------------------------------------------------===#


@always_inline
fn _mark_eviction[
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL
]() -> UInt64:
    @parameter
    if eviction_policy is CacheEviction.EVICT_NORMAL:
        return 0
    else:
        return inlined_assembly[
            "createpolicy.fractional.L2::evict_first.b64 $0;",
            UInt64,
            constraints="=l",
        ]()


@always_inline
fn async_copy[
    type: AnyType, //,
    size: Int,
    *,
    fill: Fill = Fill.NONE,
    bypass_L1_16B: Bool = True,
    l2_prefetch: OptionalReg[Int] = None,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src: UnsafePointer[type, AddressSpace.GLOBAL],
    dst: UnsafePointer[type, AddressSpace.SHARED],
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        type: The pointer type.
        size: Number of bytes to copy.
        fill: The fill to use for initializing the data.
        bypass_L1_16B: Bypass the L1 cache for 16 bypes copy.
        l2_prefetch: Enable L2 prefetching and specify the size.
        eviction_policy: Specifies the eviction policy to use.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
    """
    # TODO: Constrained on device capability.
    constrained[
        fill in (Fill.NONE, Fill.ZERO),
        "currently only zero fill is supported",
    ]()
    constrained[size in (4, 8, 16)]()
    constrained[
        not (l2_prefetch.__bool__() == bypass_L1_16B == True),
        "both enable l2 prefetching and l1 bypass cannot be True",
    ]()
    constrained[
        not l2_prefetch or l2_prefetch.value() in (64, 128, 256),
        "the l2 prefetch size must be in bounds",
    ]()

    alias cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    alias access_size = _int_to_str[size]()

    @parameter
    if l2_prefetch:
        alias cache_hint = ".L2::cache_hint" if eviction_policy is not CacheEviction.EVICT_NORMAL else ""

        alias asm = "cp.async." + cache_op + ".shared.global" + cache_hint + ".L2::" + _int_to_str[
            l2_prefetch.value()
        ]() + "B [$0], [$1], $2"

        var cache_policy = _mark_eviction[eviction_policy]()

        @parameter
        if fill is Fill.ZERO:

            @parameter
            if eviction_policy is CacheEviction.EVICT_NORMAL:
                inlined_assembly[
                    asm + ", $3;", NoneType, constraints="r,l,n,r"
                ](Int32(int(dst)), src, Int32(size), Int32(size))
            else:
                inlined_assembly[
                    asm + ", $3, $4;", NoneType, constraints="r,l,n,r,l"
                ](Int32(int(dst)), src, Int32(size), Int32(size), cache_policy)
        else:

            @parameter
            if eviction_policy is CacheEviction.EVICT_NORMAL:
                inlined_assembly[asm + ";", NoneType, constraints="r,l,n"](
                    Int32(int(dst)), src, Int32(size)
                )
            else:
                inlined_assembly[
                    asm + ", $3;", NoneType, constraints="r,l,n,l"
                ](Int32(int(dst)), src, Int32(size), cache_policy)
    else:
        constrained[
            fill is Fill.NONE, "fill is only implemented with l2_prefetch"
        ]()
        alias intrin = "llvm.nvvm.cp.async." + cache_op + ".shared.global." + access_size
        llvm_intrinsic[intrin, NoneType](dst, src)


@always_inline
fn async_copy[
    type: AnyType, //,
    size: Int,
    *,
    bypass_L1_16B: Bool = True,
](
    src: UnsafePointer[type, AddressSpace.GLOBAL, *_],
    dst: UnsafePointer[type, AddressSpace.SHARED, *_],
    src_size: Int32,
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        type: The pointer type.
        size: Number of bytes to copy.
        bypass_L1_16B: Bypass the L1 cache for 16 bypes copy.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
        src_size: Size of data transferred from src. If `src_size` < `size`,
            the remainder is filled with zero. It's undefined if `src_size`
            is greater.
    """
    # TODO: Constrained on device capability.
    constrained[size == 4 or size == 8 or size == 16]()

    alias cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    alias access_size = _int_to_str[size]()
    alias intrin = "llvm.nvvm.cp.async." + cache_op + ".shared.global." + access_size + ".s"
    llvm_intrinsic[intrin, NoneType](dst, src, src_size)


@always_inline
fn async_copy_commit_group():
    """Commits all prior initiated but uncommitted cp.async instructions into
    a cp.async-group.
    """
    llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


@always_inline
fn async_copy_wait_group(n: Int32):
    """Wait for the completion of `n` or asynchronous copy operations.

    Args:
        n: The number of pending cp.async-groups.
    """
    llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


@always_inline
fn async_copy_wait_all():
    """Wait for the completion of all commited cp.async-groups."""
    llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


@always_inline
fn external_memory[
    type: AnyTrivialRegType,
    *,
    address_space: _AddressSpace,
    alignment: Int,
    name: StringLiteral = "extern_ptr_syml",
]() -> UnsafePointer[type, address_space, alignment]:
    """Gets a pointer to dynamic shared memory.

    Parameters:
        type: The pointer's type.
        address_space: The address space used.
        alignment: The pointer's address alignment.
        name: The name of the external memory.

    Returns:
        A pointer to dynamic shared memory.
    """
    var extern_ptr_symbol = UnsafePointer[
        StaticTuple[type, 0], address_space, alignment
    ](
        __mlir_op.`pop.extern_ptr_symbol`[
            _type = UnsafePointer[
                StaticTuple[type, 0], address_space, alignment
            ]._mlir_type,
            name = name.value,
            alignment = alignment.value,
        ]()
    )
    return extern_ptr_symbol.bitcast[type]()


# ===----------------------------------------------------------------------===#
# TMA
# ===----------------------------------------------------------------------===#


@always_inline
fn fence_proxy_tensormap_generic_sys_acquire[
    type: AnyType,
](ptr: UnsafePointer[type, GPUAddressSpace.GENERIC, *_], size: Int32):
    """Acquires tensor map system's memory fence of particular size
    Args:
        ptr: Pointer to tensor map object in system's memory.
        size: The size of the object.
    """
    llvm_intrinsic[
        "llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys", NoneType
    ](ptr, size)


@always_inline
fn fence_proxy_tensormap_generic_sys_release():
    """Release tensor map system's memory fence."""
    llvm_intrinsic[
        "llvm.nvvm.fence.proxy.tensormap_generic.release.sys", NoneType
    ]()


@always_inline
fn cp_async_bulk_tensor_shared_cluster_global[
    dst_type: AnyType, mbr_type: AnyType, rank: Int
](
    dst_mem: UnsafePointer[dst_type, GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    mem_bar: UnsafePointer[mbr_type, GPUAddressSpace.SHARED],
    coords: IndexList[rank],
):
    """Initiates an asynchronous copy operation on the tensor data from global
    memory to shared memory.

    Args:
        dst_mem: Pointer to destination shared memory.
        tma_descriptor: Pointer to tensor map descriptor.
        mem_bar: A pointer to shared memory barrier.
        coords: Tile coordinates.
    """
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()

    @parameter
    if rank == 2:
        __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
            _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,2,1,0,0,0,0>}`
        ](
            to_llvm_shared_mem_ptr(dst_mem),
            to_llvm_ptr(tma_descriptor),
            to_i32(coords[0]),
            to_i32(coords[1]),
            to_llvm_shared_mem_ptr(mem_bar),
        )
    else:
        __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
            _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,1,1,0,0,0,0>}`
        ](
            to_llvm_shared_mem_ptr(dst_mem),
            to_llvm_ptr(tma_descriptor),
            to_i32(coords[0]),
            to_llvm_shared_mem_ptr(mem_bar),
        )


# ===----------------------------------------------------------------------===#
# CacheOperation
# ===----------------------------------------------------------------------===#


@value
struct CacheOperation:
    var _value: Int

    alias ALWAYS = Self(0)
    """Cache at all levels. This will be accessed again."""

    alias GLOBAL = Self(1)
    """Cache at global level."""

    alias STREAMING = Self(2)
    """Streaming, this is likely to be accessed once."""

    alias LAST_USE = Self(3)
    """Indicates the cache line will not be used again."""

    alias VOLATILE = Self(4)
    """Don't cache, and fetch again."""

    alias WRITE_BACK = Self(5)
    """Write back at all coherent levels."""

    alias WRITE_THROUGH = Self(6)
    """Write through to system memory."""

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    fn mnemonic(self) -> StringLiteral:
        if self is Self.ALWAYS:
            return "ca"
        if self is Self.GLOBAL:
            return "cg"
        if self is Self.STREAMING:
            return "cs"
        if self is Self.LAST_USE:
            return "lu"
        if self is Self.VOLATILE:
            return "cv"
        if self is Self.WRITE_BACK:
            return "wb"
        if self is Self.WRITE_THROUGH:
            return "wt"

        return "unknown cache operation"


# ===----------------------------------------------------------------------===#
# CacheEviction
# ===----------------------------------------------------------------------===#


@value
struct CacheEviction:
    var _value: Int

    alias EVICT_NORMAL = Self(0)
    """Cache data with normal eviction priority."""

    alias EVICT_FIRST = Self(1)
    """Data cached with this priority will be first in the eviction priority
    order and will likely be evicted when cache eviction is required. This
    priority is suitable for streaming data."""

    alias EVICT_LAST = Self(2)
    """Data cached with this priority will be last in the eviction priority
    order and will likely be evicted only after other data with EVICT_NORMAL
    or EVICT_FIRST eviction priotity is already evicted. This priority is
    suitable for data that should remain persistent in cache."""

    alias EVICT_UNCHANGED = Self(3)
    """Do not change eviction priority order as part of this operation."""

    alias NO_ALLOCATE = Self(4)
    """Do not allocate data to cache. This priority is suitable for streaming
    data."""

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    fn mnemonic(self) -> StringLiteral:
        if self is Self.EVICT_NORMAL:
            return "evict_normal"
        if self is Self.EVICT_FIRST:
            return "evict_first"
        if self is Self.EVICT_LAST:
            return "evict_last"
        if self is Self.EVICT_UNCHANGED:
            return "evict_unchanged"
        if self is Self.NO_ALLOCATE:
            return "no_allocate"
        return "unknown cache eviction"


# ===----------------------------------------------------------------------===#
# Fill
# ===----------------------------------------------------------------------===#


@value
struct Fill:
    var _value: Int

    alias NONE = Self(0)
    """No fill."""

    alias ZERO = Self(1)
    """Fill with zeros."""

    alias NAN = Self(2)
    """Fill with NaNs."""

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    fn __str__(self) -> String:
        if self is Self.NONE:
            return "none"
        if self is Self.ZERO:
            return "zero"
        if self is Self.NAN:
            return "nan"
        return "unknown fill"


# ===----------------------------------------------------------------------===#
# load
# ===----------------------------------------------------------------------===#


@always_inline
fn _load_impl[
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if triple_is_nvidia_cuda() else 1,
](ptr: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    constrained[
        ptr.address_space == _GPUAddressSpace.GENERIC,
        "must be global address space",
    ]()
    constrained[type.is_numeric(), "type must be numeric"]()

    @parameter
    if prefetch_size:
        constrained[prefetch_size.value() in (64, 128, 256)]()

    alias bytes_to_load = sizeof[type]() * width
    alias type_bitwidth = bitwidthof[type]()

    @parameter
    if bytes_to_load < sizeof[DType.uint32]():
        return ptr.load[width=width, alignment=alignment]()

    @parameter
    if type.is_floating_point() or type.is_signed():
        return bitcast[type, width](
            _load_impl[
                width=width,
                prefetch_size=prefetch_size,
                cache_policy=cache_policy,
                eviction_policy=eviction_policy,
                alignment=alignment,
            ](ptr.bitcast[_uint_type_of_width[type_bitwidth]()]())
        )

    @parameter
    if (
        type_bitwidth <= 16
        and sizeof[DType.uint32]() <= bytes_to_load < sizeof[DType.uint64]()
    ):
        return bitcast[type, width](
            _load_impl[
                width = (bytes_to_load // sizeof[DType.uint32]()),
                prefetch_size=prefetch_size,
                cache_policy=cache_policy,
                eviction_policy=eviction_policy,
                alignment=alignment,
            ](ptr.bitcast[DType.uint32]())
        )

    alias type_mnemonic = "u" + _int_to_str[type_bitwidth]()
    alias cache_policy_mnemonic = cache_policy.mnemonic()
    alias eviction_policy_mnemonic = (
        ".L1::" + eviction_policy.mnemonic()
    ) if eviction_policy != CacheEviction.EVICT_NORMAL else ""
    alias pretch_size_mnemonic = (
        ".L2::" + _int_to_str[prefetch_size.value()]() + "B"
    ) if prefetch_size else ""
    alias cache_operation = ".nc" if read_only else ""

    alias cache_policy_inst = (
        "" if cache_policy
        is CacheOperation.ALWAYS else ("." + cache_policy_mnemonic)
    )
    alias v_width = ("" if width == 1 else ".v" + _int_to_str[width]())

    alias instruction_name = "ld.global" + cache_policy_inst + cache_operation + eviction_policy_mnemonic + pretch_size_mnemonic + v_width + "." + type_mnemonic

    var res = SIMD[type, width]()

    @parameter
    if width == 1:
        var tmp = inlined_assembly[
            "ld.global " + cache_policy_inst + cache_operation + " $0, [$2];",
            Scalar[type],
            constraints="=r,l,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0])
        return SIMD[type, width](tmp)
    elif width == 2:
        var tmp = inlined_assembly[
            instruction_name + " {$0, $1}, [$2];",
            _RegisterPackType[Scalar[type], Scalar[type]],
            constraints="=r,=r,l,r,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0], res[1])
        return SIMD[type, width](tmp[0], tmp[1])
    elif width == 4:
        var tmp = inlined_assembly[
            instruction_name + " {$0, $1, $2, $3}, [$4];",
            _RegisterPackType[
                Scalar[type], Scalar[type], Scalar[type], Scalar[type]
            ],
            constraints="=r,=r,=r,=r,l,r,r,r,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0], res[1], res[2], res[3])
        return SIMD[type, width](tmp[0], tmp[1], tmp[2], tmp[3])

    var lhs = _load_impl[
        width = width // 2,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr)
    var rhs = _load_impl[
        width = width // 2,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr + width // 2)
    return rebind[SIMD[type, width]](lhs.join(rhs))


@always_inline
fn load[
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if triple_is_nvidia_cuda() else 1,
](ptr: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    return _load_impl[
        width=width,
        read_only=read_only,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr)


@always_inline
fn load[
    OffsetType: IntLike,
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if triple_is_nvidia_cuda() else 1,
](ptr: UnsafePointer[Scalar[type]], offset: OffsetType) -> SIMD[type, width]:
    return _load_impl[
        width=width,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr + offset)


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


fn _int_to_str[val: Int]() -> StringLiteral:
    constrained[val in (1, 2, 4, 8, 16, 32, 64, 128, 256)]()

    @parameter
    if val == 1:
        return "1"
    elif val == 2:
        return "2"
    elif val == 4:
        return "4"
    elif val == 8:
        return "8"
    elif val == 16:
        return "16"
    elif val == 32:
        return "32"
    elif val == 64:
        return "64"
    elif val == 128:
        return "128"
    elif val == 256:
        return "256"

    return "Unknown"
