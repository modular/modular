# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from collections import OptionalReg
from sys import alignof, bitwidthof, is_amd_gpu, is_gpu, is_nvidia_gpu, sizeof
from sys._assembly import inlined_assembly
from sys.intrinsics import _RegisterPackType
from sys.info import _is_sm_9x

from builtin.dtype import _uint_type_of_width
from memory import UnsafePointer
from memory.pointer import AddressSpace as _AddressSpace
from memory.pointer import _GPUAddressSpace
from memory.unsafe import bitcast

from utils import IndexList, StaticTuple
from .intrinsics import Scope
from ._utils import to_i16, to_i32, to_i64, to_llvm_ptr, to_llvm_shared_mem_ptr

# ===-----------------------------------------------------------------------===#
# AddressSpace
# ===-----------------------------------------------------------------------===#

alias AddressSpace = _GPUAddressSpace

# ===-----------------------------------------------------------------------===#
# CacheOperation
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# CacheEviction
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# Fill
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# Consistency
# ===-----------------------------------------------------------------------===#


@value
struct Consistency:
    var _value: Int

    alias WEAK = Self(0)
    """Weak consistency."""

    alias RELAXED = Self(1)
    """Relaxed consistency."""

    alias ACQUIRE = Self(2)
    """Acquire consistency."""

    alias RELEASE = Self(3)
    """Release consistency."""

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    fn __str__(self) -> String:
        return self.mnemonic()

    @always_inline
    fn mnemonic(self) -> StringLiteral:
        if self is Self.WEAK:
            return "weak"
        if self is Self.RELAXED:
            return "relaxed"
        if self is Self.ACQUIRE:
            return "acquire"
        if self is Self.RELEASE:
            return "release"

        return "unknown consistency"


# ===-----------------------------------------------------------------------===#
# ReduceOp
# ===-----------------------------------------------------------------------===#


@value
struct ReduceOp:
    var _value: Int

    alias ADD = Self(0)
    """Reduce operation: add."""

    alias MIN = Self(1)
    """Reduce operation: minimum."""

    alias MAX = Self(2)
    """Reduce operation: maximum."""

    alias AND = Self(3)
    """Reduce operation: bitwise AND."""

    alias OR = Self(4)
    """Reduce operation: bitwise OR."""

    alias XOR = Self(5)
    """Reduce operation: bitwise XOR."""

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
        return self.mnemonic()

    @always_inline
    fn mnemonic(self) -> StringLiteral:
        if self is Self.ADD:
            return "add"
        if self is Self.MIN:
            return "min"
        if self is Self.MAX:
            return "max"
        if self is Self.AND:
            return "and"
        if self is Self.OR:
            return "or"
        if self is Self.XOR:
            return "xor"

        return "unknown reduce operation"


# ===-----------------------------------------------------------------------===#
# cp.async
# ===-----------------------------------------------------------------------===#


@always_inline
fn _mark_eviction[
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL
]() -> UInt64:
    """Returns the eviction policy value for GPU cache operations.

    Parameters:
        eviction_policy: The cache eviction policy to use.

    Returns:
        Handle to the eviction policy.
    """

    @parameter
    if eviction_policy is CacheEviction.EVICT_NORMAL:
        return 0
    else:
        return inlined_assembly[
            "createpolicy.fractional.L2::evict_first.b64 $0;",
            UInt64,
            constraints="=l",
        ]()


@always_inline("nodebug")
fn async_copy[
    type: DType, //,
    size: Int,
    *,
    fill: OptionalReg[Scalar[type]] = None,
    bypass_L1_16B: Bool = True,
    l2_prefetch: OptionalReg[Int] = None,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src: UnsafePointer[Scalar[type], address_space = AddressSpace.GLOBAL],
    dst: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED],
    src_size: Int32 = 0,
    predicate: Bool = False,
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
        src_size: The size of data actually copied. When src_size < size, the
            rest is set to zero by the instruction.
        predicate: Specifies the predicate used for async_copy.
    """
    constrained[
        not fill or sizeof[type]() <= sizeof[Int32](),
        "if the fill value is specified, then the type must be 32bit or less",
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

    @parameter
    if is_amd_gpu():
        # Use sync load and stores for now
        # TODO(KERN-1249): add async memcopy to AMD
        alias n_scalars = size // sizeof[type]()
        var n_src_scalars = src_size // sizeof[type]()

        @parameter
        if fill:
            for i in range(n_src_scalars):
                dst.store(i, src.load(i))
            for i in range(n_src_scalars, n_scalars):
                dst.store(i, fill.value())
        else:

            @parameter
            for i in range(n_scalars):
                dst.store(i, src.load(i))
        return
    # Cache always: cache data in L1 first, then copy to shared memory.
    # Cache global: bypass L1 cache
    # We always do the latter.
    alias cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    alias access_size = _int_to_str[size]()

    alias cache_hint = ".L2::cache_hint" if eviction_policy is not CacheEviction.EVICT_NORMAL else ""
    alias cache_policy = _mark_eviction[eviction_policy]()

    alias l2_prefetch_substr = ".L2::" + _int_to_str[
        l2_prefetch.value()
    ]() + "B" if l2_prefetch else ""

    alias cp_async_asm = "cp.async." + cache_op + ".shared.global" + cache_hint + l2_prefetch_substr

    @parameter
    if Bool(fill) and Bool(fill.value() == 0):
        debug_assert(
            not predicate, "Predicate bit has to be set False for zero fill."
        )

        alias args_with_fill = " [$0], [$1], $2, $3"
        alias asm = cp_async_asm + args_with_fill

        @parameter
        if eviction_policy is CacheEviction.EVICT_NORMAL:
            inlined_assembly[asm + ";", NoneType, constraints="r,l,n,r"](
                Int32(Int(dst)), src, Int32(size), Int32(src_size)
            )
        else:
            inlined_assembly[asm + ", $4;", NoneType, constraints="r,l,n,r,l"](
                Int32(Int(dst)), src, Int32(size), Int32(src_size), cache_policy
            )
    elif fill:
        constrained[
            size == 16, "Non zero filling is supported only for 16B access."
        ]()

        # Pack filling values into 4B registers.
        @always_inline
        fn _i32_repr[fill: Scalar[type]]() -> Int32:
            @parameter
            if sizeof[type]() == 1:
                return bitcast[DType.int32, 1](
                    SIMD[type, 4](fill, fill, fill, fill)
                )
            elif sizeof[type]() == 2:
                return bitcast[DType.int32, 1](SIMD[type, 2](fill, fill))
            elif sizeof[type]() == 4:
                return bitcast[DType.int32](fill)

            return 0

        var fill_val = _i32_repr[fill.value()]()
        alias header_asm = "{\n.reg .pred p;\nsetp.ne.b32 p, $0, 0;\n"
        alias footer_asm = "@!p st.shared.v4.b32 [$1], {$4, $5, $6, $7};\n}\n"
        alias args_with_fill = " [$1], [$2], $3"
        alias copy_asm = header_asm + "@p " + cp_async_asm + args_with_fill

        @parameter
        if eviction_policy is CacheEviction.EVICT_NORMAL:
            inlined_assembly[
                copy_asm + ";\n" + footer_asm,
                NoneType,
                constraints="r,r,l,n,r,r,r,r",
            ](
                Int32(Int(predicate)),
                Int32(Int(dst)),
                src,
                Int32(size),
                fill_val,
                fill_val,
                fill_val,
                fill_val,
            )
        else:
            inlined_assembly[
                copy_asm + ", $8;\n" + footer_asm,
                NoneType,
                constraints="r,r,l,n,r,r,r,r,l",
            ](
                Int32(Int(predicate)),
                Int32(Int(dst)),
                src,
                Int32(size),
                fill_val,
                fill_val,
                fill_val,
                fill_val,
                cache_policy,
            )

    else:
        debug_assert(
            not predicate, "Predicate bit has to set False for no fill."
        )

        alias args = " [$0], [$1], $2"
        alias asm = cp_async_asm + args

        @parameter
        if eviction_policy is CacheEviction.EVICT_NORMAL:
            inlined_assembly[asm + ";", NoneType, constraints="r,l,n"](
                Int32(Int(dst)), src, Int32(size)
            )
        else:
            inlined_assembly[asm + ", $3;", NoneType, constraints="r,l,n,l"](
                Int32(Int(dst)), src, Int32(size), cache_policy
            )


@always_inline
fn async_copy_commit_group():
    """Commits all prior initiated but uncommitted cp.async instructions into
    a cp.async-group.
    """
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


@always_inline
fn async_copy_wait_group(n: Int32):
    """Wait for the completion of `n` or asynchronous copy operations.

    Args:
        n: The number of pending cp.async-groups.
    """
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


@always_inline
fn async_copy_wait_all():
    """Wait for the completion of all commited cp.async-groups."""
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


@always_inline
fn external_memory[
    type: AnyTrivialRegType,
    *,
    address_space: _AddressSpace,
    alignment: Int,
    name: StringLiteral = "extern_ptr_syml",
]() -> UnsafePointer[type, address_space=address_space, alignment=alignment]:
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
        StaticTuple[type, 0], address_space=address_space, alignment=alignment
    ](
        __mlir_op.`pop.extern_ptr_symbol`[
            _type = UnsafePointer[
                StaticTuple[type, 0],
                address_space=address_space,
                alignment=alignment,
            ]._mlir_type,
            name = name.value,
            alignment = alignment.value,
        ]()
    )
    return extern_ptr_symbol.bitcast[type]()


# ===-----------------------------------------------------------------------===#
# TMA
# ===-----------------------------------------------------------------------===#


@always_inline
fn fence_proxy_tensormap_generic_sys_acquire[
    type: AnyType,
](
    ptr: UnsafePointer[type, address_space = GPUAddressSpace.GENERIC, **_],
    size: Int32,
):
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
fn tma_store_fence():
    """Fence for SMEM stores for subsequent TMA STORE."""
    __mlir_op.`nvvm.fence.proxy`[
        _properties = __mlir_attr.`{ kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}`
    ]()


@always_inline
fn fence_mbarrier_init():
    """Fence that applies on the prior mbarrier.init."""
    __mlir_op.`nvvm.fence.mbarrier.init`[_type=None]()


@always_inline
fn cp_async_bulk_tensor_shared_cluster_global[
    dst_type: AnyType, mbr_type: AnyType, rank: Int
](
    dst_mem: UnsafePointer[dst_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    mem_bar: UnsafePointer[mbr_type, address_space = GPUAddressSpace.SHARED],
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


@always_inline
fn cp_async_bulk_tensor_shared_cluster_global_multicast[
    dst_type: AnyType, mbr_type: AnyType, rank: Int
](
    dst_mem: UnsafePointer[dst_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    mem_bar: UnsafePointer[mbr_type, address_space = GPUAddressSpace.SHARED],
    coords: IndexList[rank],
    multicast_mask: UInt16,
):
    """Initiates an asynchronous multicast load operation on the tensor data from global
    memory to shared memories of the cluster.

    Args:
        dst_mem: Pointer to destination shared memory.
        tma_descriptor: Pointer to tensor map descriptor.
        mem_bar: A pointer to shared memory barrier.
        coords: Tile coordinates.
        multicast_mask: An uint16 bitmask to the copy operation to specify which CTAs in a cluster will participate in the TMA multicast load.
    """
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()

    @parameter
    if rank == 2:
        __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
            _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,2,1,0,1,0,0>}`
        ](
            to_llvm_shared_mem_ptr(dst_mem),
            to_llvm_ptr(tma_descriptor),
            to_i32(coords[0]),
            to_i32(coords[1]),
            to_llvm_shared_mem_ptr(mem_bar),
            to_i16(multicast_mask),
        )
    else:
        __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
            _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,1,1,0,1,0,0>}`
        ](
            to_llvm_shared_mem_ptr(dst_mem),
            to_llvm_ptr(tma_descriptor),
            to_i32(coords[0]),
            to_llvm_shared_mem_ptr(mem_bar),
            to_i16(multicast_mask),
        )


@always_inline
fn cp_async_bulk_tensor_global_shared_cta[
    src_type: AnyType,
    rank: Int,
    /,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src_mem: UnsafePointer[src_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    coords: IndexList[rank],
):
    """Initiates an asynchronous copy operation on the tensor data from shared cta
    memory to global memory.

    Args:
        src_mem: Pointer to source shared memory.
        tma_descriptor: Pointer to tensor map descriptor.
        coords: Tile coordinates.
    """
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()

    alias cache_hint: Bool = eviction_policy is not CacheEviction.EVICT_NORMAL

    @parameter
    if rank == 2:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            eviction_policy._value,
            cache_hint,
        )
    else:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            eviction_policy._value,
            cache_hint,
        )


@always_inline
fn cp_async_bulk_tensor_reduce[
    src_type: AnyType,
    rank: Int,
    /,
    *,
    reduction_kind: StringLiteral,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src_mem: UnsafePointer[src_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    coords: IndexList[rank],
):
    """These instructions initiate an asynchronous reduction operation of tensor data
       in global memory with the tensor data in shared{::cta} memory, using ``tile`` mode.

    Args:
        src_mem: Pointer to source shared memory.
        tma_descriptor: Pointer to tensor map descriptor.
        coords: Tile coordinates.
    """
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()
    constrained[
        reduction_kind
        in ("add", "min", "max", "inc", "dec", "and", "or", "xor"),
        "reduction type " + reduction_kind + " is not supported",
    ]()
    alias cache_hint: Bool = eviction_policy is not CacheEviction.EVICT_NORMAL

    @parameter
    if rank == 2:
        llvm_intrinsic[
            "llvm.nvvm.cp.async.bulk.tensor.reduce."
            + reduction_kind
            + ".tile.2d",
            NoneType,
        ](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            UInt64(eviction_policy._value),
            cache_hint,
        )
    else:
        llvm_intrinsic[
            "llvm.nvvm.cp.async.bulk.tensor.reduce."
            + reduction_kind
            + ".tile.1d",
            NoneType,
        ](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            UInt64(eviction_policy._value),
            cache_hint,
        )


# ===-----------------------------------------------------------------------===#
# load
# ===-----------------------------------------------------------------------===#


@always_inline
fn _load_impl[
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if is_gpu() else 1,
](ptr: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    constrained[
        ptr.address_space == _GPUAddressSpace.GENERIC,
        "must be global address space",
    ]()
    constrained[type.is_numeric(), "type must be numeric"]()

    @parameter
    if is_amd_gpu():
        # TODO: KERN-1230
        constrained[read_only == False]()
        return ptr.load[width=width]()

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
            ](ptr.bitcast[Scalar[_uint_type_of_width[type_bitwidth]()]]())
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
            ](ptr.bitcast[UInt32]())
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
    alignment: Int = alignof[Scalar[type]]() if is_nvidia_gpu() else 1,
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
    OffsetType: Indexer,
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if is_nvidia_gpu() else 1,
](ptr: UnsafePointer[Scalar[type]], offset: OffsetType) -> SIMD[type, width]:
    return _load_impl[
        width=width,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr + offset)


# ===-----------------------------------------------------------------------===#
# MultiMem
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _get_multimem_ld_reduce_asm[
    type: DType,
    *,
    count: Int,
    reduction: ReduceOp,
    scope: Scope,
    consistency: Consistency,
    output_width: Int = 1,
]() -> StringLiteral:
    constrained[_is_sm_9x(), "multimem is only supported on SM90+ GPUs"]()
    constrained[type.is_floating_point(), "type must be floating point"]()
    constrained[
        type in (DType.float32, DType.float16, DType.bfloat16),
        "type must be float32, float16, or bfloat16",
    ]()

    alias ss = ".global"
    alias vec = ".v" + _int_to_str[count]()
    alias op = "." + reduction.mnemonic()
    alias type_mnemonic = "." + _get_type_mnemonic[type]() + (
        "x" + _int_to_str[output_width]() if output_width > 1 else ""
    )
    alias asm = "multimem.ld_reduce." + consistency.mnemonic() + "." + scope.mnemonic() + ss + op + vec + type_mnemonic

    return asm


@always_inline("nodebug")
fn multimem_ld_reduce[
    type: DType,
    *,
    count: Int,
    reduction: ReduceOp,
    scope: Scope,
    consistency: Consistency,
    output_width: Int = 1,
](
    addr: UnsafePointer[Scalar[type], address_space = AddressSpace.GLOBAL],
) -> StaticTuple[SIMD[type, output_width], count]:
    constrained[count in (2, 4), "count must be 2 or 4"]()

    alias asm = _get_multimem_ld_reduce_asm[
        type,
        count=count,
        reduction=reduction,
        scope=scope,
        consistency=consistency,
        output_width=output_width,
    ]()

    @parameter
    if count == 2:
        var r = inlined_assembly[
            asm + " {$0,$1}, [$2];",
            _RegisterPackType[
                SIMD[type, output_width], SIMD[type, output_width]
            ],
            constraints="=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())
        return StaticTuple[SIMD[type, output_width], count](r[0], r[1])

    @parameter
    if count == 4:
        var r = inlined_assembly[
            asm + " {$0,$1,$2,$3}, [$4];",
            _RegisterPackType[
                SIMD[type, output_width],
                SIMD[type, output_width],
                SIMD[type, output_width],
                SIMD[type, output_width],
            ],
            constraints="=r,=r,=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())

        return StaticTuple[SIMD[type, output_width], count](
            r[0], r[1], r[2], r[3]
        )

    return StaticTuple[SIMD[type, output_width], count]()


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


fn _get_type_mnemonic[type: DType]() -> StringLiteral:
    if type is DType.float32:
        return "f32"
    elif type is DType.float16:
        return "f16"
    elif type is DType.bfloat16:
        return "bf16"
    if type is DType.float64:
        return "f64"

    return "unknown type mnemonic"


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
