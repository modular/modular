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

from std.collections import Optional
from std.gpu import block_dim, lane_id, thread_idx
from std.gpu.memory import AddressSpace
from std.sys import align_of

from .coord import Idx
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

        dst_fragments.copy(src_fragments.bitcast[dst_fragments.dtype]())


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
            dst_fragments.copy(src_fragments.bitcast[dst_fragments.dtype]())
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
        dst.copy(src_fragments.bitcast[dst.dtype]())


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
        dst_fragments.copy(src.bitcast[dst_fragments.dtype]())


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
        dst.copy(src_fragments.bitcast[dst.dtype]())


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
            dst_fragments.copy(src.bitcast[dst_fragments.dtype]())
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
