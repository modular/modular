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

from std.bit import log2_floor
from std.collections import Optional
from std.gpu import block_dim, lane_id, thread_idx
from std.gpu.memory import AddressSpace, CacheEviction, async_copy
from std.sys import align_of, size_of

from layout import Idx
from .layout_tensor import ThreadScope
from .swizzle import Swizzle
from .tile_layout import Layout
from .tile_tensor import TileTensor


@always_inline("nodebug")
def _get_worker_idx[thread_scope: ThreadScope]() -> Int:
    """Returns the worker index for the current thread scope.

    Returns the index of the current worker (thread) based on the specified
    thread scope. For `BLOCK` scope, returns the thread's flat index within
    the (up to 3D) thread block. For `WARP` scope, returns the lane ID
    within the warp.

    Duplicated from `layout_tensor._get_worker_idx` to avoid a cross-module
    import; the BLOCK path unconditionally uses the 3D formula, since for
    1D/2D launches `thread_idx.z == 0` and `block_dim.y/z == 1` and the
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

    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` into `dst`.

        Both tensors must share the same `element_size` so the copy
        operates on matching logical element widths.

        Parameters:
            element_size: Number of scalar elements per logical element.

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

    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Asynchronously copies `src` into `dst`.

        The copy may not be complete when this call returns; callers
        must commit and synchronize before reading `dst`.

        Parameters:
            element_size: Number of scalar elements per logical element.

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

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` in generic memory into `dst` in shared memory.

        Masked bounds checking is not supported.

        Parameters:
            element_size: Number of scalar elements per logical element.

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

        dst_fragments.copy_from(src_fragments.bitcast[dst_fragments.dtype]())


@fieldwise_init
struct SharedToGenericTileCopier[
    thread_layout: Layout,
    *,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
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
    """

    comptime src_address_space = AddressSpace.SHARED
    """Source `AddressSpace` this copier reads from."""
    comptime dst_address_space = AddressSpace.GENERIC
    """Destination `AddressSpace` this copier writes to."""

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` in shared memory into `dst` in generic memory.

        The non-swizzled path uses `TileTensor.copy`, which widens to SIMD
        stores when the layouts permit. The swizzled path walks per-thread
        elements explicitly and applies the swizzle to the source fragment
        offsets.

        Masked bounds checking, fp32 -> half precision downcast, and
        `binary_op` fusion are not supported.

        Parameters:
            element_size: Number of scalar elements per logical element.

        Args:
            dst: Destination tile in generic memory.
            src: Source tile in shared memory.
        """
        comptime assert dst.dtype == src.dtype, "src and dst dtype must match."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[ThreadScope.BLOCK]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        comptime if not Self.swizzle:
            dst_fragments.copy_from(
                src_fragments.bitcast[dst_fragments.dtype]()
            )
        else:
            comptime simd_size = src.element_size
            comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
            comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()
            comptime swizzle_fn = Self.swizzle.value()

            var src_frag_offset = src_fragments._distance(src.ptr)
            comptime num_stores_per_thread = (
                src_fragments.LayoutType.static_product // simd_size
            )

            comptime for i in range(num_stores_per_thread):
                var src_idx = src_fragments.layout(Idx[i * simd_size]())
                var dst_idx = dst_fragments.layout(Idx[i * simd_size]())
                var src_idx_base = src_idx % Int64(swizzle_fn.size())
                var src_idx_diff = src_idx - src_idx_base
                var swizzled_idx = swizzle_fn(
                    Scalar[src.linear_idx_type](src_frag_offset)
                    + Scalar[src.linear_idx_type](src_idx_base)
                ) + Scalar[src.linear_idx_type](src_idx_diff)

                var src_vec = src.ptr.load[
                    width=simd_size, alignment=src_align
                ](swizzled_idx).cast[dst.dtype]()
                dst_fragments.ptr.mut_cast[True]().store[alignment=dst_align](
                    dst_idx, src_vec
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

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` in generic memory into `dst` in local memory.

        Masked bounds checking is not supported.

        Parameters:
            element_size: Number of scalar elements per logical element.

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

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` in local memory into `dst` in generic memory.

        Masked bounds checking is not supported.

        Parameters:
            element_size: Number of scalar elements per logical element.

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

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` in shared memory into `dst` in local memory.

        Parameters:
            element_size: Number of scalar elements per logical element.

        Args:
            dst: Destination tile in local memory.
            src: Source tile in shared memory.
        """
        comptime assert (
            dst.dtype == src.dtype
        ), "dst dtype must be the same as src dtype."

        var worker_idx = _get_worker_idx[Self.thread_scope]()
        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
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

    The AMD `row_major` prefetch pattern and fp32 -> half precision
    downcast are not supported.

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

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Copies `src` in local memory into `dst` in shared memory.

        Parameters:
            element_size: Number of scalar elements per logical element.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in local memory.
        """
        comptime assert src.dtype == dst.dtype, "src and dst dtype must match."

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        comptime if not Self.swizzle:
            dst_fragments.copy_from(src.bitcast[dst_fragments.dtype]())
        else:
            comptime simd_size = src.element_size
            comptime src_align = align_of[SIMD[src.dtype, simd_size]]()
            comptime dst_align = align_of[SIMD[dst.dtype, simd_size]]()
            comptime swizzle_fn = Self.swizzle.value()

            var dst_frag_offset = dst_fragments._distance(dst.ptr)
            comptime num_vecs = src.LayoutType.static_product // simd_size

            comptime for i in range(num_vecs):
                var src_idx = src.layout(Idx[i * simd_size]())
                var dst_idx = dst_fragments.layout(Idx[i * simd_size]())
                var dst_idx_base = dst_idx % Int64(swizzle_fn.size())
                var dst_idx_diff = dst_idx - dst_idx_base
                var swizzled_idx = swizzle_fn(
                    Scalar[dst.linear_idx_type](dst_frag_offset)
                    + Scalar[dst.linear_idx_type](dst_idx_base)
                ) + Scalar[dst.linear_idx_type](dst_idx_diff)
                var src_vec = src.ptr.load[
                    width=simd_size, alignment=src_align
                ](src_idx).cast[dst.dtype]()
                dst.ptr.mut_cast[True]().store[alignment=dst_align](
                    swizzled_idx, src_vec
                )


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

    @always_inline("nodebug")
    def copy[
        element_size: Int
    ](
        self,
        dst: TileTensor[
            mut=True,
            address_space=Self.dst_address_space,
            element_size=element_size,
            ...,
        ],
        src: TileTensor[
            address_space=Self.src_address_space,
            element_size=element_size,
            ...,
        ],
    ):
        """Asynchronously copies `src` in generic memory into `dst` in shared
        memory.

        The copy is issued via `cp.async` on NVIDIA. Callers must commit
        and wait on the copy before using the destination tile.

        Parameters:
            element_size: Number of scalar elements per logical element.

        Args:
            dst: Destination tile in shared memory.
            src: Source tile in generic memory.
        """
        comptime assert (
            src.dtype == dst.dtype
        ), "src dtype and dst dtype must be the same."

        comptime element_size_bytes = size_of[src.dtype]() * element_size
        comptime assert element_size_bytes in (
            4,
            8,
            16,
        ), "async copy only supports 4, 8, or 16 byte vector elements."

        # The swizzle's `base` parameter sets how many least-significant
        # bits of the offset are kept constant. cp.async requires the
        # destination address to be aligned to `element_size` scalars,
        # so the swizzle must not permute bits below `log2(element_size)`.
        # `make_swizzle[..., access_size=element_size]` constructs a
        # swizzle that satisfies this (it sets `base = log2_floor(access_size)`);
        # a hand-rolled `Swizzle(bits, base, shift)` with
        # `base < log2_floor(element_size)` would silently produce
        # misaligned offsets and trigger CUDA_ERROR_MISALIGNED_ADDRESS.
        comptime if Self.swizzle:
            comptime assert Self.swizzle.value().base >= log2_floor(
                element_size
            ), (
                "swizzle.base is too small for the requested element_size:"
                " cp.async would receive misaligned offsets. Construct the"
                " swizzle with `make_swizzle[..., access_size=element_size]`"
                " (or a hand-rolled `Swizzle(bits, base, shift)` with"
                " `base >= log2_floor(element_size)`)."
            )

        comptime num_busy_threads = Self.thread_layout.size()
        var worker_idx = _get_worker_idx[Self.thread_scope]()

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var src_fragments = src.distribute[Self.thread_layout](worker_idx)
        var dst_fragments = dst.distribute[Self.thread_layout](worker_idx)

        # The trailing `bitcast` materializes the pointee type to a concrete
        # `Scalar[src.dtype]` so `async_copy`'s `dtype` parameter infers
        # cleanly; without it the inferred dtype is a comptime expression
        # that fails to unify across the two pointer arguments.
        comptime dtype = src.dtype
        var src_global_ptr = (
            src_fragments.ptr.address_space_cast[AddressSpace.GLOBAL]()
            .mut_cast[False]()
            .unsafe_origin_cast[ImmutAnyOrigin]()
            .bitcast[Scalar[dtype]]()
        )
        var dst_shared_ptr = (
            dst_fragments.ptr.mut_cast[True]()
            .address_space_cast[AddressSpace.SHARED]()
            .unsafe_origin_cast[MutAnyOrigin]()
            .bitcast[Scalar[dtype]]()
        )

        # Per-thread fragments are sized in logical (post-vectorize) elements,
        # so `static_product` already counts cp.async issues, not scalars: each
        # iteration issues one `element_size_bytes`-wide async copy covering
        # `element_size` scalars. For non-vectorized inputs (`element_size ==
        # 1`) this degenerates to one scalar-sized async copy per element.
        comptime num_issues = src_fragments.LayoutType.static_product

        # For masked copies we bound `src_idx` by the absolute end of the
        # valid src region: `src.dim[0]() * row_stride - src_frag_offset`.
        # `row_stride` is the leading-dim stride; for row-major tiles this
        # equals the column count. The bound is computed in `Int` to keep
        # the comparison free of scalar-dtype unification fights.
        comptime row_stride_static = src.LayoutType.static_stride[0]
        var src_frag_offset = Int(src_fragments._distance(src.ptr))
        var src_idx_bound = (
            Int(src.dim[0]()) * Int(row_stride_static) - src_frag_offset
        )

        # When swizzling, the destination address is computed in absolute
        # tile coordinates (relative to `dst.ptr`), then rebased back to the
        # fragment by subtracting `dst_frag_offset`. The unswizzled path uses
        # the per-fragment offset directly.
        comptime if Self.swizzle:
            comptime swizzle_fn = Self.swizzle.value()
            var dst_frag_offset = dst_fragments._distance(dst.ptr)
            var dst_frag_offset_typed = Scalar[dst.linear_idx_type](
                dst_frag_offset
            )
            comptime for i in range(num_issues):
                var src_idx = Int(src_fragments.layout(Idx[i]()))
                var dst_idx_raw = dst_fragments.layout(Idx[i]())
                var dst_idx_base = dst_idx_raw % Int64(swizzle_fn.size())
                var dst_idx_diff = dst_idx_raw - dst_idx_base
                var swizzled_idx = (
                    swizzle_fn(
                        dst_frag_offset_typed
                        + Scalar[dst.linear_idx_type](dst_idx_base)
                    )
                    + Scalar[dst.linear_idx_type](dst_idx_diff)
                    - dst_frag_offset_typed
                )

                comptime if Self.masked:
                    var src_copy_size = Int32(element_size_bytes) if (
                        src_idx < src_idx_bound
                    ) else Int32(0)
                    async_copy[
                        element_size_bytes,
                        fill=Scalar[src.dtype](0),
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + Int(swizzled_idx),
                        src_copy_size,
                    )
                else:
                    async_copy[
                        element_size_bytes,
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + Int(swizzled_idx),
                    )
        else:
            comptime for i in range(num_issues):
                var src_idx = Int(src_fragments.layout(Idx[i]()))
                var dst_idx = Int(dst_fragments.layout(Idx[i]()))

                comptime if Self.masked:
                    var src_copy_size = Int32(element_size_bytes) if (
                        src_idx < src_idx_bound
                    ) else Int32(0)
                    async_copy[
                        element_size_bytes,
                        fill=Scalar[src.dtype](0),
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + dst_idx,
                        src_copy_size,
                    )
                else:
                    async_copy[
                        element_size_bytes,
                        eviction_policy=Self.eviction_policy,
                    ](
                        src_global_ptr + src_idx,
                        dst_shared_ptr + dst_idx,
                    )
