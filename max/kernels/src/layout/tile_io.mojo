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
"""Trait and utilities for copying data between `TileTensor`s."""

from std.collections import Optional, OptionalReg
from max.gpu import block_dim, lane_id, thread_idx
from max.gpu.memory import CacheEviction, Fill
from std.math.uutils import umod
from std.sys import align_of
from std.os import abort

from layout import Coord, Idx
from .swizzle import Swizzle
from .tile_layout import Layout, row_major
from .tile_tensor import TileTensor
from .tensor_engine import DefaultEngine


struct ThreadScope(EnumLike, TrivialRegisterPassable, Writable):
    """Represents the scope of thread operations in GPU programming.

    This struct defines the scope at which thread operations are performed,
    particularly for operations like tensor distribution and synchronization.
    It provides two main scopes: `BLOCK` and `WARP`, which correspond to
    different levels of thread grouping in GPU programming models.

    Example:

    ```mojo
    from layout.tile_io import copy_dram_to_sram, ThreadScope

    # Distribute tensor at block level (all threads in block participate)
    copy_dram_to_sram[layout, thread_scope=ThreadScope.BLOCK](dst, src)

    # Distribute tensor at warp level (only threads in same warp participate)
    copy_dram_to_sram[layout, thread_scope=ThreadScope.WARP](dst, src)
    ```

    Performance:

    - WARP scope operations typically have lower synchronization overhead
        than BLOCK scope operations.
    - BLOCK scope operations allow coordination across all threads in a block,
        which is necessary for certain algorithms.
    - The choice of scope can significantly impact performance and correctness
        of parallel algorithms.

    Notes:

    - The appropriate scope depends on the specific algorithm and hardware.
    - WARP scope operations may be more efficient for operations that only
        require coordination within a warp.
    - BLOCK scope operations are necessary when threads from different warps
        need to coordinate.
    - The actual size of a warp or block is hardware-dependent.
    """

    var _value: Int32
    """The internal integer value representing the thread scope."""

    comptime BLOCK = Self(0)
    """Represents operations at the thread block level, where all threads in a
    block participate."""

    comptime WARP = Self(1)
    """Represents operations at the warp level, where only threads within the
    same warp participate."""

    comptime _enum_case_names = ParameterList.of[
        "BLOCK".value,
        "WARP".value,
    ].values

    comptime _enum_case_types = TypeList.splat[
        ParameterList[Self._enum_case_names].length, NoneType
    ].values

    @inline(.always)
    def _get_enum_discriminant(self) -> Int:
        return Int(self._value)

    @inline(.always)
    def _unsafe_get_enum_payload[
        id: Int
    ](ref self) -> ref[self] TypeList[Trait=AnyType, Self._enum_case_types]()[
        id
    ]:
        comptime assert False, "ThreadScope has no payload"

    def __init__(out self, value: Int):
        """Initialize a `ThreadScope` with the given integer value.

        Args:
            value: An integer representing the thread scope (0 for `BLOCK`,
                1 for `WARP`).
        """
        self._value = Int32(value)

    def __eq__(self, other: Self) -> Bool:
        """Compare two `ThreadScope` objects for equality.

        Args:
            other: Another `ThreadScope` object to compare with.

        Returns:
            True if the thread scopes are equal, False otherwise.
        """
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        """Compare two `ThreadScope` objects for inequality.

        Args:
            other: Another `ThreadScope` object to compare with.

        Returns:
            True if the thread scopes are not equal, False otherwise.
        """
        return not (self == other)

    def write_to(self, mut writer: Some[Writer]):
        """Write the `ThreadScope` as a human-readable string representation.

        Args:
            writer: The writer to write the string representation to.

        Aborts:
            If the thread scope has an invalid value.
        """
        if self == Self.BLOCK:
            return writer.write("BLOCK")
        if self == Self.WARP:
            return writer.write("WARP")
        abort("invalid ThreadScope entry")

    def __int__(self) -> Int:
        """Convert the `ThreadScope` to an integer value.

        Returns:
            The integer value of the thread scope (0 for BLOCK, 1 for WARP).
        """
        return Int(self._value)


@inline(.nodebug)
def _get_worker_idx[thread_scope: ThreadScope]() -> Int:
    """Returns the worker index for the current thread scope.

    Returns the index of the current worker (thread) based on the specified
    thread scope. For `BLOCK` scope, returns the thread's flat index within
    the (up to 3D) thread block. For `WARP` scope, returns the lane ID
    within the warp.

    The BLOCK path unconditionally uses the 3D formula, since for 1D/2D
    launches `thread_idx.z == 0` and `block_dim.y/z == 1` and the
    extra terms fold away at compile time.

    Parameters:
        thread_scope: The scope at which the worker index is determined.

    Returns:
        The worker index within the specified scope.
    """
    comptime if thread_scope == ThreadScope.BLOCK:
        return (
            thread_idx.z * block_dim.y * block_dim.x
            + thread_idx.y * block_dim.x
            + thread_idx.x
        )
    else:
        return lane_id()


trait TileCopier:
    """Trait for copying a `TileTensor` from one address space to another.

    Implementors move a source `TileTensor` in `src_address_space` into a
    destination `TileTensor` in `dst_address_space`. The two address
    spaces are advertised as compile-time fields on the implementor so
    callers can introspect what a given copier is wired to do.
    """

    comptime src_address_space: AddressSpace
    """Source `AddressSpace` the copier reads from."""

    comptime dst_address_space: AddressSpace
    """Destination `AddressSpace` the copier writes to."""

    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` into `dst`.

        Both tensors must share the same `element_size` so the copy
        operates on matching logical element widths.

        Args:
            dst: Destination tile in `dst_address_space`.
            src: Source tile in `src_address_space`.
        """
        ...


trait AsyncTileCopier:
    """Trait for asynchronously copying a `TileTensor` between address spaces.

    Distinct from `TileCopier` because async copies have semantics the
    synchronous trait cannot express: on NVIDIA the `copy` call only
    issues the transfer, and callers must commit it via
    `async_copy_commit_group()` and synchronize via
    `async_copy_wait_all()` or `async_copy_wait_group()` before reading
    the destination tile. Keeping the trait separate prevents code
    generic over `TileCopier` from silently accepting an async copier
    and producing reads that race the in-flight transfer, and gives the
    async path room to grow (e.g., explicit commit/wait members or
    multi-stage pipelining) as the async story is fleshed out.
    """

    comptime src_address_space: AddressSpace
    """Source `AddressSpace` the copier reads from."""

    comptime dst_address_space: AddressSpace
    """Destination `AddressSpace` the copier writes to."""

    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Asynchronously copies `src` into `dst`.

        The copy may not be complete when this call returns; callers
        must commit and synchronize before reading `dst`.

        Args:
            dst: Destination tile in `dst_address_space`.
            src: Source tile in `src_address_space`.
        """
        ...


@fieldwise_init
struct GenericToSharedTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from generic memory into shared memory.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Optional swizzle applied to the shared-memory destination
            for bank-conflict mitigation. `None` produces a straight copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed
            (`BLOCK` or `WARP`). Defaults to `ThreadScope.BLOCK`.
    """

    comptime src_address_space = AddressSpace.GENERIC
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.SHARED
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in generic memory into `dst` in shared memory.

        Masked bounds checking is not supported.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[
            Self.thread_layout, swizzle=Self.swizzle
        ](worker_idx)

        comptime assert src.element_size == dst.element_size
        dst_fragments.copy_from(src_fragments.bitcast[dst_fragments.dtype]())


@fieldwise_init
struct SharedToGenericTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    masked: Bool = False,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from shared memory into generic memory.

    The `swizzle` parameter is a property of the shared-memory tile being
    read and must match the swizzle used when that tile was written;
    passing a mismatched (or `None`) swizzle produces incorrect data.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Swizzle the shared-memory tile was populated with.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        masked: When `True`, vectors whose destination offset falls at or past
            `valid_rows * row_stride` are not stored, so the valid row count
            may be smaller than the source tile. Defaults to `False`, which
            keeps the unmasked behavior. Independently of `masked`, the
            source may be FP32 with an FP16 or BF16 destination.
    """

    comptime src_address_space = AddressSpace.SHARED
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.GENERIC
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in shared memory into `dst` in generic memory.

        When `masked` is set, the valid-row bound is `dst.dim[0]()`. For a
        destination whose row dim is static (e.g. a `.tile[...]` sub-view),
        use `copy_bounded` with an explicit runtime row count instead.

        Args:
            dst: Destination tile in generic memory.
            src: Source tile in shared memory.
        """
        self.copy_bounded(dst, src, None)

    @inline(.nodebug)
    def copy_bounded(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
        dst_num_valid_rows: OptionalReg[Int],
    ):
        """Copies `src` in shared memory into `dst` in generic memory with an
        optional explicit valid-row bound.

        The non-swizzled, same-dtype, unmasked path uses `TileTensor.copy`,
        which widens to SIMD stores when the layouts permit. Otherwise the
        copy walks per-thread vectors explicitly, applying the swizzle to the
        source offsets, an FP32 to FP16/BF16 downcast, and (when `masked`)
        a per-vector bound check mirroring the legacy `copy_sram_to_dram`.
        `binary_op` fusion is not supported.

        Args:
            dst: Destination tile in generic memory.
            src: Source tile in shared memory.
            dst_num_valid_rows: Valid-row count for the masked bound. When
                `None`, `dst.dim[0]()` is used. Only consulted when `masked`
                is `True`. It overrides `dst.dim[0]()` so the destination can
                keep a static shape (and therefore the same thread
                distribution as the source) while the row count is dynamic.
        """
        comptime assert dst.dtype == src.dtype or (
            src.dtype == .float32
            and (dst.dtype == .float16 or dst.dtype == .bfloat16)
        ), "Only matching dtypes or FP32 to FP16/BF16 are supported."
        comptime assert src.element_size == dst.element_size

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[ThreadScope.BLOCK]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        comptime if Self.masked or src.dtype != dst.dtype:
            comptime simd_size = src.element_size
            comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
            comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()
            comptime num_stores_per_thread = (
                src_fragments.LayoutType.static_product
            )

            var src_frag_offset = Scalar[src.linear_idx_type](
                src_fragments._distance(src)
            )
            # Bound on the fragment-relative destination offset: the absolute
            # end of the valid rows minus the fragment's offset in `dst`.
            var dst_idx_bound = Scalar[dst.linear_idx_type]()
            comptime if Self.masked:
                var dst_num_rows = dst_num_valid_rows.or_else(Int(dst.dim[0]()))
                dst_idx_bound = Scalar[dst.linear_idx_type](
                    dst_num_rows * Int(dst.dynamic_stride(0))
                ) - Scalar[dst.linear_idx_type](dst_fragments._distance(dst))

            var src_flat = src.reshape(
                row_major(Int(src.layout(src.layout.product() - 1)) + 1)
            )
            var dst_frag_flat = dst_fragments.reshape(
                row_major(
                    Int(
                        dst_fragments.layout(dst_fragments.layout.product() - 1)
                    )
                    + 1
                )
            )

            comptime for i in range(num_stores_per_thread):
                var src_idx = src_fragments.layout[
                    linear_idx_type=src.linear_idx_type
                ](Idx[i])
                var dst_idx = dst_fragments.layout[
                    linear_idx_type=dst.linear_idx_type
                ](Idx[i])

                comptime if case .Some(swizzle_fn) = Self.swizzle:
                    var src_idx_base = umod(
                        src_idx,
                        Scalar[src.linear_idx_type](swizzle_fn.size()),
                    )
                    var src_idx_diff = src_idx - src_idx_base
                    src_idx = swizzle_fn(
                        src_frag_offset + src_idx_base
                    ) + Scalar[src.linear_idx_type](src_idx_diff)
                else:
                    src_idx = src_frag_offset + src_idx

                var in_bounds = True
                comptime if Self.masked:
                    in_bounds = dst_idx < dst_idx_bound
                if in_bounds:
                    var src_vec = src_flat.load[alignment=src_align](
                        Coord(src_idx)
                    ).cast[dst.dtype]()
                    dst_frag_flat.store[alignment=dst_align](
                        Coord(dst_idx), src_vec
                    )
        else:
            comptime if case .Some(swizzle_fn) = Self.swizzle:
                comptime simd_size = src.element_size
                comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
                comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()

                var src_frag_offset = Scalar[src.linear_idx_type](
                    src_fragments._distance(src)
                )
                comptime num_stores_per_thread = (
                    src_fragments.LayoutType.static_product
                )
                var src_flat = src.reshape(
                    row_major(Int(src.layout(src.layout.product() - 1)) + 1)
                )
                var dst_frag_flat = dst_fragments.reshape(
                    row_major(
                        Int(
                            dst_fragments.layout(
                                dst_fragments.layout.product() - 1
                            )
                        )
                        + 1
                    )
                )

                comptime for i in range(num_stores_per_thread):
                    var src_idx = src_fragments.layout[
                        linear_idx_type=src.linear_idx_type
                    ](Idx[i])
                    var dst_idx = dst_fragments.layout(Idx[i])
                    var src_idx_base = umod(
                        src_idx,
                        Scalar[src.linear_idx_type](swizzle_fn.size()),
                    )
                    var src_idx_diff = src_idx - src_idx_base
                    var swizzled_idx = swizzle_fn(
                        src_frag_offset + src_idx_base
                    ) + Scalar[src.linear_idx_type](src_idx_diff)

                    var src_vec = src_flat.load[alignment=src_align](
                        Coord(swizzled_idx)
                    ).cast[dst.dtype]()
                    dst_frag_flat.store[alignment=dst_align](
                        Coord(dst_idx), src_vec
                    )
            else:
                dst_fragments.copy_from(
                    src_fragments.bitcast[dst_fragments.dtype]()
                )


@fieldwise_init
struct GenericToLocalTileCopier[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from generic memory into registers.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.GENERIC
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.LOCAL
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in generic memory into `dst` in local memory.

        Masked bounds checking is not supported.

        Args:
            dst: Destination tile in local memory.
            src: Source tile in generic memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        comptime assert src.element_size == dst.element_size
        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        dst.copy_from(src_fragments.bitcast[dst.dtype]())


@fieldwise_init
struct LocalToGenericTileCopier[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from registers into generic memory.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.LOCAL
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.GENERIC
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in local memory into `dst` in generic memory.

        Masked bounds checking is not supported.

        Args:
            dst: Destination tile in generic memory.
            src: Source tile in local memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        comptime assert src.element_size == dst.element_size
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)
        dst_fragments.copy_from(src.bitcast[dst_fragments.dtype]())


@fieldwise_init
struct SharedToLocalTileCopier[
    thread_layout: Layout,
    *,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from shared memory into registers.

    `thread_layout` is used as the warp layout. `axis`-based distribution
    is not yet supported, and this copier currently only produces correct
    data when `src` was populated without a swizzle; reading a swizzled
    shared-memory tile into local memory is not yet supported.

    Parameters:
        thread_layout: Warp layout describing how threads are organized
            over the copy.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.SHARED
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.LOCAL
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in shared memory into `dst` in local memory.

        Args:
            dst: Destination tile in local memory.
            src: Source tile in shared memory.
        """
        comptime assert (
            dst.dtype == src.dtype
        ), "dst dtype must be the same as src dtype."

        var worker_idx = _get_worker_idx[Self.thread_scope]()
        var src_fragments = src.distribute[Self.thread_layout](worker_idx)

        comptime assert src.element_size == dst.element_size
        dst.copy_from(src_fragments.bitcast[dst.dtype]())


@fieldwise_init
struct LocalToSharedTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](ImplicitlyCopyable, Movable, TileCopier):
    """A `TileCopier` that moves a tile from registers into shared memory.

    Supports numeric FP32 to FP16 or BF16 conversion. The AMD `row_major`
    prefetch pattern is not supported.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Optional swizzle applied to the shared-memory
            destination; the same swizzle must be used by any subsequent
            reader of the tile.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.
    """

    comptime src_address_space = AddressSpace.LOCAL
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.SHARED
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Copies `src` in local memory into `dst` in shared memory.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in local memory.
        """
        comptime assert src.dtype == dst.dtype or (
            src.dtype == .float32
            and (dst.dtype == .float16 or dst.dtype == .bfloat16)
        ), "Only matching dtypes or FP32 to FP16/BF16 are supported."
        comptime assert src.element_size == dst.element_size

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        comptime if src.dtype != dst.dtype:
            comptime assert src.LayoutType.all_dims_known
            comptime assert dst_fragments.LayoutType.all_dims_known
            comptime assert (
                src.LayoutType.static_product
                == dst_fragments.LayoutType.static_product
            )

        comptime if case .Some(swizzle_fn) = Self.swizzle:
            comptime simd_size = src.element_size
            comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
            comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()

            var dst_frag_offset = Scalar[dst.linear_idx_type](
                dst_fragments._distance(dst)
            )
            comptime num_vecs = src.LayoutType.static_product

            var dst_flat = dst.reshape(
                row_major(Int(dst.layout(dst.layout.product() - 1)) + 1)
            )

            comptime for i in range(num_vecs):
                var dst_idx = dst_fragments.layout[
                    linear_idx_type=dst.linear_idx_type
                ](Idx[i])
                var dst_idx_base = umod(
                    dst_idx,
                    Scalar[dst.linear_idx_type](swizzle_fn.size()),
                )
                var dst_idx_diff = dst_idx - dst_idx_base
                var swizzled_idx = swizzle_fn(
                    dst_frag_offset + dst_idx_base
                ) + Scalar[dst.linear_idx_type](dst_idx_diff)
                var src_vec = src.load[alignment=src_align]((Idx[i],)).cast[
                    dst.dtype
                ]()
                dst_flat.store[alignment=dst_align](
                    Coord(swizzled_idx), src_vec
                )
        else:
            comptime if src.dtype == dst.dtype:
                dst_fragments.copy_from(src.bitcast[dst_fragments.dtype]())
            else:
                dst_fragments.copy_from(src)


@fieldwise_init
struct GenericToSharedAsyncTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    masked: Bool = False,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](AsyncTileCopier, ImplicitlyCopyable, Movable):
    """An `AsyncTileCopier` that asynchronously moves a tile from generic
    memory into shared memory using NVIDIA's `cp.async` instruction.

    On NVIDIA GPUs (compute capability 8.0+), the copy issues `cp.async`
    instructions, allowing the transfer to overlap with subsequent compute.
    On AMD and Apple GPUs the underlying `async_copy` intrinsic falls back
    to synchronous loads and stores.

    The copy is asynchronous on NVIDIA: callers must commit it via
    `async_copy_commit_group()` and synchronize via `async_copy_wait_all()`
    or `async_copy_wait_group()` before reading the destination tile.

    The vector size in bytes (`size_of[dtype]() * element_size`) must be
    4, 8, or 16.

    Parameters:
        thread_layout: Layout describing how threads are organized over
            the copy.
        swizzle: Optional swizzle applied to the shared-memory destination
            for bank-conflict mitigation. `None` produces a straight copy.
            Subsequent readers of the tile must use the same swizzle.
        masked: When `True`, performs per-vector bounds-checking against
            `src.dim[0]() * row_stride`. Vectors that fall past the bound
            issue zero-byte `cp.async` operations with `fill=0`, which the
            hardware fulfills by zeroing the destination bytes. Intended
            for source tiles whose row count is dynamic (e.g. attention
            prefill loading the tail of a sequence).
        eviction_policy: Cache eviction policy for the source data.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed
            (`BLOCK` or `WARP`). Defaults to `ThreadScope.BLOCK`.
    """

    comptime src_address_space = AddressSpace.GENERIC
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.SHARED
    """Destination `AddressSpace` this copier writes to."""

    @inline(.nodebug)
    def copy(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
    ):
        """Asynchronously copies `src` in generic memory into `dst` in shared
        memory.

        The copy is issued via `cp.async` on NVIDIA. Callers must commit
        and wait on the copy before using the destination tile.

        This satisfies the `AsyncTileCopier` trait; the masked bound is
        derived from `src.dim[0]()`. For an explicit-bound copy (a `src`
        whose row dim is static), call `copy_bounded` directly.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
        """
        self.copy_bounded(dst, src, None)

    @inline(.nodebug)
    def copy_bounded(
        self,
        dst: TileTensor[mut=True, address_space=Self.dst_address_space, ...],
        src: TileTensor[address_space=Self.src_address_space, ...],
        src_num_valid_rows: OptionalReg[Int],
    ):
        """Asynchronously copies `src` into `dst` with an optional explicit
        masked-bound override.

        Identical to `copy` except for the masked-bound source. This is NOT a
        trait method (the `AsyncTileCopier` trait fixes the `copy` signature);
        it is the explicit-bound entry point.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
            src_num_valid_rows: Explicit valid-row count for the masked bound.
                When `None`, the masked bound is derived from `src.dim[0]()`
                (byte-identical to the legacy behavior). When provided, it
                overrides `src.dim[0]()` in the
                `src_idx_bound = rows * row_stride - src_frag_offset`
                computation; everything else is unchanged. This lets callers
                whose `src` carries a static row dim (e.g. a `TileTensor.tile`
                sub-view, which does not runtime-clip dim0) still drive a
                correct partial-tile zero-fill by passing the runtime clip
                directly. Only consulted when `masked` is `True`.
        """
        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        # For masked copies we bound `src_idx` by the absolute end of the
        # valid src region: `src.dim[0]() * row_stride - src_frag_offset`.
        # `row_stride` is the leading-dim stride; for row-major tiles this
        # equals the column count.
        comptime idx_type = type_of(src_fragments).linear_idx_type
        var src_idx_bound = Scalar[idx_type]()
        comptime if Self.masked:
            # The valid-row count drives the masked zero-fill bound. By default
            # it is `src.dim[0]()` (legacy behavior). A caller may override it
            # with an explicit runtime value when `src`'s own dim0 is static
            # (e.g. a `.tile[...]` sub-view, which does not runtime-clip dim0
            # the way the legacy iterator did).
            var src_num_rows = src_num_valid_rows.or_else(Int(src.dim[0]()))
            src_idx_bound = Scalar[idx_type](
                src_num_rows * Int(src.dynamic_stride(0))
            ) - Scalar[idx_type](src_fragments._distance(src))

        # When swizzling, `copy_from_async` computes the destination address in
        # absolute tile coordinates and rebases it back onto the fragment, so
        # it needs the fragment's offset from the tile base.
        var base_offset = Scalar[type_of(dst_fragments).linear_idx_type]()
        comptime if Self.swizzle:
            base_offset = dst_fragments._distance(dst)

        dst_fragments.copy_from_async[
            is_masked=Self.masked,
            swizzle=Self.swizzle,
            fill=Fill.ZERO,
            eviction_policy=Self.eviction_policy,
        ](
            src_fragments,
            src_idx_bound=src_idx_bound,
            base_offset=base_offset,
        )


# ===----------------------------------------------------------------------=== #
# Free-function wrappers
# ===----------------------------------------------------------------------=== #
#
# Thin module-level wrappers that delegate to the `TileCopier` structs above.
# They mirror the names (and the common-case parameter sets) of the legacy
# free `copy_*` functions so that migrating callers off the legacy tensor type
# requires only a type change at the call site, not a rename.
#
# These intentionally cover only the common case. The following variants are
# deliberately deferred as follow-ups (they have no `TileCopier` backing yet,
# or require behavior the structs do not implement):
#   - All AMD `buffer_load`/`buffer_store` overloads (the
#     `src_iter`/`bound`/`dst_base`/`src_base`/`offset`/`cache_policy`
#     parameter family).
#   - `copy_local_to_local` (no corresponding `TileCopier` exists).
#   - `binary_op` fusion on the SRAM -> DRAM path.
#   - fp32 -> half-precision downcast.
#   - The `axis` (SRAM -> LOCAL), `row_major` (LOCAL -> SHARED), and
#     `fill`/mask parameters.
#   - `block_dim_count`: the legacy functions used it to recover a flat
#     worker index across multi-dimensional block launches. `_get_worker_idx`
#     (used by every `TileCopier` here) already computes the flat 3D index
#     unconditionally, so the common single-launch case needs no equivalent
#     parameter; multi-block-tile launches that relied on `block_dim_count`
#     are left to the follow-up that ports those callers.


@inline(.nodebug)
def copy_dram_to_sram[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.GENERIC, ...],
):
    """Synchronously copies a tile from DRAM (generic memory) to SRAM (shared).

    Delegates to `GenericToSharedTileCopier`.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Optional swizzle applied to the shared-memory destination for
            bank-conflict mitigation. `None` produces a straight copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in shared memory.
        src: Source tile in generic memory.
    """
    GenericToSharedTileCopier[
        thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@inline(.nodebug)
def copy_sram_to_dram[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    masked: Bool = False,
](
    dst: TileTensor[mut=True, address_space=.GENERIC, ...],
    src: TileTensor[address_space=.SHARED, ...],
    dst_num_valid_rows: OptionalReg[Int] = None,
):
    """Synchronously copies a tile from SRAM (shared memory) to DRAM (generic).

    Delegates to `SharedToGenericTileCopier`. The `binary_op` fusion path of
    the legacy free function is not supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Swizzle the shared-memory tile was populated with; must match
            the swizzle used when the tile was written.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        masked: When `True`, vectors past `valid_rows * row_stride` are not
            stored, where `valid_rows` is `dst_num_valid_rows` or, when that
            is `None`, `dst.dim[0]()`. An FP32 source may be downcast to an
            FP16 or BF16 destination regardless of `masked`.

    Args:
        dst: Destination tile in generic memory.
        src: Source tile in shared memory.
        dst_num_valid_rows: Explicit valid-row count for the masked bound;
            lets `dst` keep a static shape (and thus the same thread
            distribution as `src`) with a runtime row count.
    """
    SharedToGenericTileCopier[
        thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        masked=masked,
    ]().copy_bounded(dst, src, dst_num_valid_rows)


@inline(.nodebug)
def copy_local_to_dram[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.GENERIC, ...],
    src: TileTensor[address_space=.LOCAL, ...],
):
    """Synchronously copies a tile from registers (LOCAL) to DRAM (generic).

    Delegates to `LocalToGenericTileCopier`. The AMD `buffer_store` path of
    the legacy free function is not supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in generic memory.
        src: Source tile in local memory.
    """
    LocalToGenericTileCopier[
        thread_layout,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@inline(.nodebug)
def copy_local_to_dram[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.GENERIC, ...],
    src: TileTensor[address_space=.LOCAL, ...],
    valid_rows: Int,
):
    """Copies register vectors into the valid rows of a generic-memory tile.

    Clamps the signed row bound to the destination height. Unlike the
    unbounded overload, this path uses natural scalar alignment for vector
    loads and stores, including when every row is valid.

    Parameters:
        thread_layout: Canonical row-major, flat rank-two thread layout.
        num_threads: Number of participating threads, with the same filtering
            semantics as the unbounded overload.
        thread_scope: Scope used to obtain the worker index.

    Args:
        dst: Flat rank-two FP32 tile with contiguous two-scalar vectors.
        src: Static FP32 register layout with the same distributed vector count.
        valid_rows: Number of valid destination rows, relative to `dst`.

    Constraints:
        Uses `DefaultEngine` with vector width two. Destination dimensions
        must be positive, static, and divisible by the thread dimensions.
        Source scalar offsets must describe disjoint contiguous vectors.
        Destination column stride must be two; its positive row stride must
        cover the entire row. Bases and scalar offsets require FP32 alignment.
        Allocations and signed index types must accommodate the full layout.
        BLOCK workers that wrap the thread layout require distinct destination
        views and register owners, as in independent warp copies.
    """
    comptime assert src.dtype == .float32 and dst.dtype == .float32
    comptime assert src.Engine == DefaultEngine[element_width=2]
    comptime assert dst.Engine == DefaultEngine[element_width=2]
    comptime assert src.rank == 2 and src.flat_rank == 2
    comptime assert dst.rank == 2 and dst.flat_rank == 2
    comptime assert src.LayoutType.all_dims_known
    comptime assert dst.LayoutType.shape_known
    comptime assert thread_layout.rank == 2
    comptime assert thread_layout.shape_known and thread_layout.stride_known
    comptime rows = dst.LayoutType.static_shape[0]
    comptime cols = dst.LayoutType.static_shape[1]
    comptime thread_rows = thread_layout.static_shape[0]
    comptime thread_cols = thread_layout.static_shape[1]
    comptime assert rows > 0 and cols > 0
    comptime assert thread_rows > 0 and thread_cols > 0
    comptime assert thread_layout.static_stride[0] == thread_cols
    comptime assert thread_layout.static_stride[1] == 1
    comptime assert rows % thread_rows == 0 and cols % thread_cols == 0
    comptime assert dst.LayoutType.static_stride[1] == 2
    comptime assert src.LayoutType.static_product == (
        rows * cols // thread_layout.size()
    )
    debug_assert(Int(dst.dynamic_stride(0)) >= 2 * cols)

    var row_bound = min(rows, max(0, valid_rows))
    if row_bound == 0:
        return
    var worker_idx = _get_worker_idx[thread_scope]()
    comptime if num_threads > thread_layout.size():
        if worker_idx >= thread_layout.size():
            return

    var thread_row = (worker_idx // thread_cols) % thread_rows
    var thread_col = worker_idx % thread_cols
    comptime fragment_rows = rows // thread_rows
    comptime for i in range(src.LayoutType.static_product):
        var row = thread_row + (i % fragment_rows) * thread_rows
        if row < row_bound:
            var col = thread_col + (i // fragment_rows) * thread_cols
            dst._store_storage[alignment=4](
                dst.layout[linear_idx_type=dst.linear_idx_type](
                    Coord(row, col)
                ),
                rebind[dst.ElementType](src[(i)]),
            )


@inline(.nodebug)
def copy_dram_to_local[
    thread_layout: Layout,
    *,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.LOCAL, ...],
    src: TileTensor[address_space=.GENERIC, ...],
):
    """Synchronously copies a tile from DRAM (generic memory) to registers.

    Delegates to `GenericToLocalTileCopier`. The AMD `buffer_load` path of the
    legacy free function is not supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in local memory.
        src: Source tile in generic memory.
    """
    GenericToLocalTileCopier[
        thread_layout,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@inline(.nodebug)
def copy_local_to_shared[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.LOCAL, ...],
):
    """Synchronously copies a tile from registers (LOCAL) to SRAM (shared).

    Delegates to `LocalToSharedTileCopier`, including numeric FP32 to FP16
    or BF16 conversion. The legacy AMD `row_major` prefetch pattern is not
    supported here.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Optional swizzle applied to the shared-memory destination;
            the same swizzle must be used by any subsequent reader of the tile.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in shared memory.
        src: Source tile in local memory.
    """
    LocalToSharedTileCopier[
        thread_layout,
        swizzle=swizzle,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@inline(.nodebug)
def copy_sram_to_local[
    thread_layout: Layout,
    *,
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.LOCAL, ...],
    src: TileTensor[address_space=.SHARED, ...],
):
    """Synchronously copies a tile from SRAM (shared memory) to registers.

    Delegates to `SharedToLocalTileCopier`. The `axis`-based distribution of
    the legacy free function is not supported here.

    Parameters:
        thread_layout: Warp layout describing how threads are organized over
            the copy.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in local memory.
        src: Source tile in shared memory.
    """
    SharedToLocalTileCopier[
        thread_layout,
        thread_scope=thread_scope,
    ]().copy(dst, src)


@inline(.nodebug)
def copy_dram_to_sram_async[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    masked: Bool = False,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    num_threads: Int = thread_layout.size(),
    thread_scope: ThreadScope = ThreadScope.BLOCK,
](
    dst: TileTensor[mut=True, address_space=.SHARED, ...],
    src: TileTensor[address_space=.GENERIC, ...],
    src_num_valid_rows: OptionalReg[Int] = None,
):
    """Asynchronously copies a tile from DRAM (generic memory) to SRAM (shared).

    Delegates to `GenericToSharedAsyncTileCopier`, which issues NVIDIA
    `cp.async` instructions (falling back to synchronous loads/stores on AMD
    and Apple GPUs). The copy is asynchronous: callers must commit it via
    `async_copy_commit_group()` and synchronize via `async_copy_wait_all()`
    or `async_copy_wait_group()` before reading the destination tile.

    Unlike the legacy free function, whose `swizzle: Bool` auto-derived an
    ldmatrix swizzle, this wrapper takes an `Optional[Swizzle]` directly and
    does not replicate that auto-derivation; pass an explicit swizzle (for
    example from `make_swizzle[..., access_size=element_size]`) when one is
    required.

    Parameters:
        thread_layout: Layout describing how threads are organized over the
            copy.
        swizzle: Optional swizzle applied to the shared-memory destination for
            bank-conflict mitigation. `None` produces a straight copy.
            Subsequent readers of the tile must use the same swizzle.
        masked: When `True`, performs per-vector bounds-checking; vectors past
            the bound issue zero-filling `cp.async` operations.
        eviction_policy: Cache eviction policy for the source data.
        num_threads: Total number of threads in the thread block. Threads
            beyond `thread_layout.size()` do not participate.
        thread_scope: Scope at which thread operations are performed.

    Args:
        dst: Destination tile in shared memory.
        src: Source tile in generic memory.
        src_num_valid_rows: Explicit valid-row count for the masked bound.
            When `None` (default) the bound is derived from `src.dim[0]()`
            (byte-identical to legacy). When provided it overrides
            `src.dim[0]()` so a `src` with a static row dim (e.g. a
            `TileTensor.tile` sub-view) can still drive a correct partial-tile
            zero-fill. Only consulted when `masked` is `True`.
    """
    GenericToSharedAsyncTileCopier[
        thread_layout,
        swizzle=swizzle,
        masked=masked,
        eviction_policy=eviction_policy,
        num_threads=num_threads,
        thread_scope=thread_scope,
    ]().copy_bounded(dst, src, src_num_valid_rows)
