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
"""Tests for `OwnedTensor` backed by `OwnedGPUEngine` shared-memory storage.

`OwnedGPUStorage` gets its buffer from `stack_allocation`, which only outlives
the constructor in shared memory, so every check runs inside a kernel.
"""

from layout import Coord, Idx, TensorLayout, TileTensor, col_major, row_major
from layout.owned_tensor import (
    OwnedGPUEngine,
    OwnedGPUStorage,
    OwnedGPUTensor,
    OwnedTensor,
    OwnedTensorEngine,
)
from layout.tensor_engine import DefaultEngine, TensorEngine
from layout.tile_layout import Layout

from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from std.testing import assert_equal


# Non-square so a swapped or symmetric decode of the extents is caught.
comptime _ROWS = 4
comptime _COLS = 8
comptime _NUM_ELEMENTS = _ROWS * _COLS


def reverse_through_shared_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var owned = OwnedGPUTensor[.float32, .SHARED](row_major[_ROWS, _COLS]())
    var flat = TileTensor(owned).reshape(row_major[_NUM_ELEMENTS]())

    var tid = Int(thread_idx.x)
    flat[tid] = Float32(tid)
    barrier()
    # Reading another thread's write only works if every thread's
    # `OwnedTensor` resolved to the same shared allocation.
    output.unsafe_store(tid, flat[_NUM_ELEMENTS - 1 - tid][0])


def two_tensors_kernel(output: Pointer[Float32, MutAnyOrigin]):
    # Same type and layout, so both `stack_allocation` calls instantiate
    # identically; each must still get its own shared buffer.
    var first = OwnedGPUTensor[.float32, .SHARED](row_major[_ROWS, _COLS]())
    var second = OwnedGPUTensor[.float32, .SHARED](row_major[_ROWS, _COLS]())
    var first_flat = TileTensor(first).reshape(row_major[_NUM_ELEMENTS]())
    var second_flat = TileTensor(second).reshape(row_major[_NUM_ELEMENTS]())

    var tid = Int(thread_idx.x)
    first_flat[tid] = Float32(tid)
    second_flat[tid] = Float32(_NUM_ELEMENTS + tid)
    barrier()
    # If the buffers alias, `second`'s writes overwrite `first`'s.
    output.unsafe_store(tid, first_flat[_NUM_ELEMENTS - 1 - tid][0])
    output.unsafe_store(
        _NUM_ELEMENTS + tid, second_flat[_NUM_ELEMENTS - 1 - tid][0]
    )


def storage_view_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var owned = OwnedGPUTensor[.float32, .SHARED](row_major[_ROWS, _COLS]())
    var view = OwnedGPUEngine.get_storage_view[result_origin=origin_of(owned)](
        owned._storage
    )
    var tile = TileTensor(owned)

    output.unsafe_store(
        0, Float32(Int(view) - Int(owned._storage.unsafe_ptr()))
    )
    output.unsafe_store(1, Float32(Int(view) - Int(tile.unsafe_ptr())))

    view.unsafe_store(3, Float32(42))
    output.unsafe_store(2, tile.reshape(row_major[_NUM_ELEMENTS]())[3][0])

    output.unsafe_store(3, Float32(Int(tile.dim[0]())))
    output.unsafe_store(4, Float32(Int(tile.dim[1]())))


def tile_type_and_layout_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var owned = OwnedGPUTensor[.float32, .SHARED](col_major[_ROWS, _COLS]())
    var tile = TileTensor(owned)
    comptime assert (
        type_of(tile)
        == TileTensor[
            .float32,
            type_of(owned).LayoutType,
            origin_of(owned),
            address_space=.SHARED,
        ]
    )

    output.unsafe_store(0, Float32(Int(tile.dim[0]())))
    output.unsafe_store(1, Float32(Int(tile.dim[1]())))
    output.unsafe_store(2, Float32(Int(tile.dynamic_stride(0))))
    output.unsafe_store(3, Float32(Int(tile.dynamic_stride(1))))


def col_major_indexing_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var owned = OwnedGPUTensor[.float32, .SHARED](col_major[_ROWS, _COLS]())
    var tile = TileTensor(owned)

    var tid = Int(thread_idx.x)
    var row = tid // _COLS
    var col = tid % _COLS
    tile[row, col] = Float32(row * _COLS + col)
    barrier()
    output.unsafe_store(tid, tile.unsafe_ptr().unsafe_load(tid))


def implicit_conversion_kernel(output: Pointer[Float32, MutAnyOrigin]):
    var owned = OwnedGPUTensor[.float32, .SHARED](row_major[_ROWS, _COLS]())
    var implicit_tile: TileTensor[
        .float32,
        type_of(owned).LayoutType,
        origin_of(owned),
        address_space=.SHARED,
    ] = owned
    implicit_tile[1, 2] = Float32(7)
    output.unsafe_store(0, TileTensor(owned)[1, 2][0])


comptime _WIDTH = 2


struct _PairedGPUEngine(OwnedTensorEngine):
    """Stores `_WIDTH` scalars per layout element and views them as SIMD.

    Its `ViewEngine` differs from the `TileTensor` default, so a `TileTensor`
    built from it only indexes correctly if the constructor takes the owner's
    `ViewEngine`.
    """

    comptime StorageType[
        dtype: DType, address_space: AddressSpace, LayoutType: TensorLayout
    ]: Deinitable & Defaultable = OwnedGPUStorage[
        dtype,
        address_space,
        type_of(row_major[LayoutType.static_product * _WIDTH]()),
    ]

    comptime ViewEngine: TensorEngine = DefaultEngine[element_width=_WIDTH]

    @staticmethod
    def get_storage_view[
        result_mut: Bool,
        dtype: DType,
        address_space: AddressSpace,
        result_origin: Origin[mut=result_mut],
        //,
    ](
        ref storage: Self.StorageType[dtype, address_space, _],
    ) -> Self.ViewEngine.StorageType[dtype, result_origin, address_space]:
        return (
            storage.unsafe_ptr()
            .unsafe_bitcast[SIMD[dtype, _WIDTH]]()
            .unsafe_mut_cast[result_mut]()
            .unsafe_origin_cast[result_origin]()
        )


def view_engine_kernel(output: Pointer[Float32, MutAnyOrigin]):
    # Vectorized engines index in scalars, so strides are scaled by `_WIDTH`,
    # as `TileTensor.vectorize` does.
    var layout = Layout(
        Coord(Idx[_ROWS], Idx[_COLS]),
        Coord(Idx[_COLS * _WIDTH], Idx[_WIDTH]),
    )
    var owned = OwnedTensor[
        .float32, .SHARED, type_of(layout), Engine=_PairedGPUEngine
    ](layout)
    var tile = TileTensor(owned)
    comptime assert type_of(tile).Engine == DefaultEngine[element_width=_WIDTH]

    for row in range(_ROWS):
        for col in range(_COLS):
            var base = Float32(row * _COLS + col) * _WIDTH
            tile[row, col] = SIMD[.float32, _WIDTH](base, base + 1)

    var scalars = tile.unsafe_ptr()
    for i in range(_NUM_ELEMENTS * _WIDTH):
        output.unsafe_store(i, scalars.unsafe_load(i))


def test_shared_storage_is_visible_across_threads(ctx: DeviceContext) raises:
    print("== test_shared_storage_is_visible_across_threads")
    var buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[reverse_through_shared_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=_NUM_ELEMENTS
    )

    var host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    for i in range(_NUM_ELEMENTS):
        assert_equal(host[i], Float32(_NUM_ELEMENTS - 1 - i))


def test_two_tensors_get_distinct_storage(ctx: DeviceContext) raises:
    print("== test_two_tensors_get_distinct_storage")
    comptime count = 2 * _NUM_ELEMENTS
    var buf = ctx.enqueue_create_buffer[.float32](count)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[two_tensors_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=_NUM_ELEMENTS
    )

    var host = ctx.enqueue_create_host_buffer[.float32](count)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    for i in range(_NUM_ELEMENTS):
        assert_equal(host[i], Float32(_NUM_ELEMENTS - 1 - i))
        assert_equal(
            host[_NUM_ELEMENTS + i], Float32(2 * _NUM_ELEMENTS - 1 - i)
        )


def test_storage_view_aliases_tile_tensor(ctx: DeviceContext) raises:
    print("== test_storage_view_aliases_tile_tensor")
    var buf = ctx.enqueue_create_buffer[.float32](5)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[storage_view_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )

    var host = ctx.enqueue_create_host_buffer[.float32](5)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    assert_equal(host[0], Float32(0), "view differs from storage pointer")
    assert_equal(host[1], Float32(0), "view differs from TileTensor pointer")
    assert_equal(host[2], Float32(42), "write through view not seen by tile")
    assert_equal(host[3], Float32(_ROWS))
    assert_equal(host[4], Float32(_COLS))


def test_tile_keeps_owner_type_and_layout(ctx: DeviceContext) raises:
    print("== test_tile_keeps_owner_type_and_layout")
    var buf = ctx.enqueue_create_buffer[.float32](4)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[tile_type_and_layout_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )

    var host = ctx.enqueue_create_host_buffer[.float32](4)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    assert_equal(host[0], Float32(_ROWS))
    assert_equal(host[1], Float32(_COLS))
    assert_equal(host[2], Float32(1))
    assert_equal(host[3], Float32(_ROWS))


def test_tile_indexes_through_owner_layout(ctx: DeviceContext) raises:
    print("== test_tile_indexes_through_owner_layout")
    var buf = ctx.enqueue_create_buffer[.float32](_NUM_ELEMENTS)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[col_major_indexing_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=_NUM_ELEMENTS
    )

    var host = ctx.enqueue_create_host_buffer[.float32](_NUM_ELEMENTS)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    # Memory offset `i` of a column-major tile holds element
    # `(i % _ROWS, i // _ROWS)`, which the kernel set to `row * _COLS + col`.
    for i in range(_NUM_ELEMENTS):
        assert_equal(host[i], Float32((i % _ROWS) * _COLS + i // _ROWS))


def test_implicit_conversion_aliases_owner(ctx: DeviceContext) raises:
    print("== test_implicit_conversion_aliases_owner")
    var buf = ctx.enqueue_create_buffer[.float32](1)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[implicit_conversion_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )

    var host = ctx.enqueue_create_host_buffer[.float32](1)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    assert_equal(host[0], Float32(7))


def test_tile_uses_owner_view_engine(ctx: DeviceContext) raises:
    print("== test_tile_uses_owner_view_engine")
    comptime count = _NUM_ELEMENTS * _WIDTH
    var buf = ctx.enqueue_create_buffer[.float32](count)
    buf.enqueue_fill(Float32(-1))

    ctx.enqueue_function[view_engine_kernel](
        buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )

    var host = ctx.enqueue_create_host_buffer[.float32](count)
    ctx.enqueue_copy(host, buf)
    ctx.synchronize()

    # Each layout element is `_WIDTH` consecutive scalars, so the scalars
    # come out as a plain ramp.
    for i in range(count):
        assert_equal(host[i], Float32(i))


def main() raises:
    with DeviceContext() as ctx:
        test_shared_storage_is_visible_across_threads(ctx)
        test_two_tensors_get_distinct_storage(ctx)
        test_storage_view_aliases_tile_tensor(ctx)
        test_tile_keeps_owner_type_and_layout(ctx)
        test_tile_indexes_through_owner_layout(ctx)
        test_implicit_conversion_aliases_owner(ctx)
        test_tile_uses_owner_view_engine(ctx)
