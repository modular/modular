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

from kv_cache.types import (
    KVCacheStaticParams,
    PagedKVCache,
    PagedKVCacheCollection,
)
from layout import (
    DefaultEngine,
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from std.memory import alloc
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.testing import assert_true

from std.utils.coord import dyn_coord
from std.utils.index import IndexList
from std.collections import OptionalReg

comptime kv_params = KVCacheStaticParams(num_heads=16, head_size=16)
comptime _rows_view_kv_params = KVCacheStaticParams(num_heads=1, head_size=4)
comptime _rows_view_page_size = 1
comptime _rows_view_granularity = 4

# The one scaled collection type the scales test and the rows-view tests
# build; a second spelling would drift from it.
comptime _ScaledCollection = PagedKVCacheCollection[
    .float32,
    _rows_view_kv_params,
    _rows_view_page_size,
    MutAnyOrigin,
    ImmutAnyOrigin,
    ImmutAnyOrigin,
    MutAnyOrigin,
    scale_dtype_=DType.float32,
    quantization_granularity_=_rows_view_granularity,
]


def _zeroed_blocks_pool[num_blocks: Int]() -> _ScaledCollection.blocks_tt_type:
    """Allocates a zeroed value pool that outlives the caller.

    The collection keeps a view of the pool, so it must be heap-backed;
    a local `List` would free at return.
    """
    comptime shape = IndexList[6](
        num_blocks,
        2,
        1,
        _rows_view_page_size,
        _rows_view_kv_params.num_heads,
        _rows_view_kv_params.head_size,
    )
    var ptr = unsafe_alloc[Float32](shape.flattened_length())
    return rebind[_ScaledCollection.blocks_tt_type](
        TileTensor(ptr, row_major(shape.flattened_length()))
        .reshape(
            Coord(
                Int64(shape[0]),
                Idx[2],
                Int64(shape[2]),
                Idx[_rows_view_page_size],
                Idx[_rows_view_kv_params.num_heads],
                Idx[_rows_view_kv_params.head_size],
            )
        )
        .as_unsafe_any_origin()
    ).fill(Float32(0))


def _uninit_scales_pool[
    num_scale_blocks: Int
]() -> _ScaledCollection.scales_tt_type:
    """Allocates an uninitialized scale pool that outlives the caller.

    The collection keeps a view of the pool, so it must be heap-backed;
    each caller fills the values it asserts on.
    """
    comptime shape = IndexList[6](
        num_scale_blocks,
        2,
        1,
        _rows_view_page_size,
        _rows_view_kv_params.num_heads,
        1,
    )
    var ptr = unsafe_alloc[Float32](shape.flattened_length())
    return rebind[_ScaledCollection.scales_tt_type](
        TileTensor(ptr, row_major(shape.flattened_length()))
        .reshape(
            Coord(
                Int64(shape[0]),
                Idx[2],
                Int64(shape[2]),
                Idx[_rows_view_page_size],
                Idx[_rows_view_kv_params.num_heads],
                Idx[1],
            )
        )
        .as_unsafe_any_origin()
    )


def do_test[
    page_size: Int,
    layout_block_size: Int,
    scale_dtype: Optional[DType] = None,
]() raises:
    comptime batch_size = 16
    comptime max_num_blocks = 100
    comptime shape = IndexList[6](
        100,
        2,
        1,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )

    var blocks_ptr = List(length=shape.flattened_length(), fill=Float32(0))
    comptime Collection = PagedKVCacheCollection[
        .float32,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=scale_dtype,
    ]
    var blocks = TileTensor(
        Span(blocks_ptr), row_major(shape.flattened_length())
    ).reshape(
        Coord(
            Int64(shape[0]),
            Idx[2],
            Int64(shape[2]),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        )
    )
    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(0))
    var cache_lengths = TileTensor(
        Span(cache_lengths_ptr), row_major(len(cache_lengths_ptr))
    ).reshape(Coord(Int64(batch_size)))
    var lookup_table_ptr = List(
        length=batch_size * max_num_blocks, fill=UInt32(0)
    )
    var lookup_table = TileTensor(
        Span(lookup_table_ptr), row_major(len(lookup_table_ptr))
    ).reshape(Coord(Int64(batch_size), Int64(max_num_blocks)))
    for i in range(batch_size):
        cache_lengths[i] = UInt32(i)
        for j in range(max_num_blocks):
            lookup_table[i, j] = UInt32(j)

    var max_seq_length = UInt32(2048)
    var max_cache_length = UInt32(2048)

    comptime scales_dtype = scale_dtype.or_else(.float32)
    var scales: OptionalReg[Collection.scales_tt_type] = None

    comptime if scale_dtype == DType.float8_e4m3fn:
        # Use the same shape as the blocks.
        var scales_ptr = unsafe_alloc[Scalar[scales_dtype]](
            shape.flattened_length()
        )
        scales = rebind[Collection.scales_tt_type](
            TileTensor(scales_ptr, row_major(shape.flattened_length()))
            .reshape(
                Coord(
                    Int64(shape[0]),
                    Idx[2],
                    Int64(shape[2]),
                    Idx[page_size],
                    Idx[kv_params.num_heads],
                    Idx[kv_params.head_size],
                )
            )
            .as_unsafe_any_origin()
        ).fill(0)

    var collection = Collection(
        rebind[Collection.blocks_tt_type](blocks.as_unsafe_any_origin()),
        cache_lengths.as_imm().as_unsafe_any_origin(),
        lookup_table.as_imm().as_unsafe_any_origin(),
        max_seq_length,
        max_cache_length,
        scales,
    )

    var cache = collection.get_key_cache(1)
    _ = cache.block_paged_ptr[layout_block_size](1, layout_block_size, 0)


def test_paged_kv_cache_stride_is_runtime() raises:
    """Test that PagedKVCache keeps stride[0] runtime for view correctness.

    PagedKVCache is a 4D view of a 6D parent tensor. The outer stride depends
    on num_layers from the parent tensor, which is unknown at compile time.
    Keeping stride[0] a runtime (Int64) leaf ensures offset calculations use
    the runtime stride rather than an incorrect compile-time value.

    This is a regression test - previously the row-major layout computed
    stride[0] from just the 4D shape, which was incorrect for view tensors.
    """
    comptime CacheType = PagedKVCache[
        .float32,
        kv_params,
        16,
        MutUntrackedOrigin,
        DefaultEngine[element_width=1],
        ImmUntrackedOrigin,
        DefaultEngine[element_width=1],
        ImmUntrackedOrigin,
        DefaultEngine[element_width=1],
        MutUntrackedOrigin,
        DefaultEngine[element_width=1],
    ]

    # Verify stride[0] is a runtime leaf, not a `ComptimeInt`.
    comptime stride_0_is_static = CacheType.blocks_tt_layout.stride_types[
        0
    ].is_static_value
    assert_true(
        not stride_0_is_static,
        "PagedKVCache stride[0] should be a runtime stride",
    )

    # Verify inner strides are still known (enables partial constant folding)
    comptime stride_1 = CacheType.blocks_tt_layout.stride_types[1].static_value
    comptime stride_2 = CacheType.blocks_tt_layout.stride_types[2].static_value
    comptime stride_3 = CacheType.blocks_tt_layout.stride_types[3].static_value

    comptime expected_stride_1 = kv_params.num_heads * kv_params.head_size
    comptime expected_stride_2 = kv_params.head_size
    comptime expected_stride_3 = 1

    assert_true(
        stride_1 == expected_stride_1,
        String("PagedKVCache stride[1] should be ")
        + String(expected_stride_1)
        + ", got "
        + String(stride_1),
    )
    assert_true(
        stride_2 == expected_stride_2,
        String("PagedKVCache stride[2] should be ")
        + String(expected_stride_2)
        + ", got "
        + String(stride_2),
    )
    assert_true(
        stride_3 == expected_stride_3,
        String("PagedKVCache stride[3] should be ")
        + String(expected_stride_3)
        + ", got "
        + String(stride_3),
    )


def test_paged_kv_cache_offset_correctness() raises:
    """Test that PagedKVCache offset calculations use correct runtime strides.

    This test verifies that when accessing elements through a PagedKVCache view,
    the correct values from the underlying 6D tensor are returned. This catches
    bugs where compile-time strides don't match the actual memory layout.

    The 6D parent tensor has shape:
        [num_blocks, 2, num_layers, page_size, num_heads, head_size]

    The 4D PagedKVCache view has shape:
        [num_blocks, page_size, num_heads, head_size]

    The view's stride[0] must account for the skipped dimensions (2, num_layers).
    """
    comptime num_blocks = 4
    comptime num_layers = 3
    comptime page_size = 2
    comptime num_heads = 2
    comptime head_size = 4

    comptime kv_params_small = KVCacheStaticParams(
        num_heads=num_heads, head_size=head_size
    )

    # 6D shape: [num_blocks, 2, num_layers, page_size, num_heads, head_size]
    comptime shape_6d = IndexList[6](
        num_blocks, 2, num_layers, page_size, num_heads, head_size
    )
    comptime total_elems = shape_6d.flattened_length()

    # Allocate and fill with unique values (value = flattened index)
    var blocks_ptr = List(length=total_elems, fill=Float32(0))
    for i in range(total_elems):
        blocks_ptr[i] = Float32(i)

    comptime Collection = PagedKVCacheCollection[
        .float32,
        kv_params_small,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var blocks = TileTensor(
        Span(blocks_ptr), row_major(shape_6d.flattened_length())
    ).reshape(
        Coord(
            Int64(shape_6d[0]),
            Idx[2],
            Int64(shape_6d[2]),
            Idx[page_size],
            Idx[kv_params_small.num_heads],
            Idx[kv_params_small.head_size],
        )
    )

    # Create minimal supporting tensors
    comptime batch_size = 1
    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(0))
    var cache_lengths = TileTensor(
        Span(cache_lengths_ptr), row_major(len(cache_lengths_ptr))
    ).reshape(Coord(Int64(batch_size)))

    var lookup_table_ptr = List(length=batch_size * num_blocks, fill=UInt32(0))
    for i in range(num_blocks):
        lookup_table_ptr[i] = UInt32(i)  # Identity mapping
    var lookup_table = TileTensor(
        Span(lookup_table_ptr), row_major(len(lookup_table_ptr))
    ).reshape(Coord(Int64(batch_size), Int64(num_blocks)))

    # Create collection
    var collection = Collection(
        rebind[Collection.blocks_tt_type](blocks.as_unsafe_any_origin()),
        cache_lengths.as_imm().as_unsafe_any_origin(),
        lookup_table.as_imm().as_unsafe_any_origin(),
        UInt32(page_size),
        UInt32(page_size),
    )

    # Test: Access element through key cache view's blocks tensor and verify correct value
    # Get key cache for layer 1 (kv_idx=0, layer_idx=1)
    var key_cache = collection.get_key_cache(1)

    # 4D coords [block=1, page=0, head=1, dim=2]
    # In 6D, this corresponds to [block=1, kv=0, layer=1, page=0, head=1, dim=2]
    #
    # 6D strides (row-major):
    #   stride[0] = 2 * 3 * 2 * 2 * 4 = 96
    #   stride[1] = 3 * 2 * 2 * 4 = 48
    #   stride[2] = 2 * 2 * 4 = 16
    #   stride[3] = 2 * 4 = 8
    #   stride[4] = 4
    #   stride[5] = 1
    #
    # Expected 6D offset = 1*96 + 0*48 + 1*16 + 0*8 + 1*4 + 2*1 = 96 + 16 + 4 + 2 = 118
    comptime expected_6d_offset = (
        1 * (2 * num_layers * page_size * num_heads * head_size)
        + 0 * (num_layers * page_size * num_heads * head_size)
        + 1 * (page_size * num_heads * head_size)
        + 0 * (num_heads * head_size)
        + 1 * head_size
        + 2
    )

    # The 4D view's runtime stride[0] should be 2 * num_layers * page_size * num_heads * head_size
    # If the bug existed (using compile-time stride[0] = page_size * num_heads * head_size),
    # we'd compute wrong offset: 1*16 + 0*8 + 1*4 + 2 = 22 (wrong!)

    # Access via the blocks TileTensor - this tests the layout offset computation
    var idx = dyn_coord[.int64]((1, 0, 1, 2))
    var value = key_cache.blocks.load(idx)
    var expected_value = Float32(expected_6d_offset)

    assert_true(
        value == expected_value,
        String("PagedKVCache returned wrong value! Got ")
        + String(value)
        + " but expected "
        + String(expected_value)
        + " (at 6D offset "
        + String(expected_6d_offset)
        + "). This indicates stride[0] is using incorrect compile-time value.",
    )


def test_paged_kv_cache_quantization() raises:
    comptime CacheType = PagedKVCache[
        .float32,
        kv_params,
        16,
        MutUntrackedOrigin,
        DefaultEngine[element_width=1],
        ImmUntrackedOrigin,
        DefaultEngine[element_width=1],
        ImmUntrackedOrigin,
        DefaultEngine[element_width=1],
        MutUntrackedOrigin,
        DefaultEngine[element_width=1],
        scale_dtype_=DType.float8_e4m3fn,
        quantization_granularity_=256,
    ]
    assert_true(CacheType.quantization_enabled, "Quantization not enabled")
    assert_true(
        CacheType.quantization_granularity == 256,
        "Incorrect quantization granularity",
    )


def test_scales_resolve_through_their_own_lookup_table() raises:
    """A distinct scales_lookup_table must resolve scale pages independently
    of lookup_table, not silently fall back to it.
    """
    comptime batch_size = 1
    comptime num_value_blocks = 1
    comptime num_scale_blocks = 3

    var blocks = _zeroed_blocks_pool[num_value_blocks]()

    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(1))
    var cache_lengths = TileTensor(
        Span(cache_lengths_ptr), row_major(len(cache_lengths_ptr))
    ).reshape(Coord(Int64(batch_size)))

    # Values resolve through block 0.
    var lookup_table_ptr = unsafe_alloc[UInt32](batch_size)
    lookup_table_ptr[0] = UInt32(0)
    var lookup_table = (
        TileTensor(lookup_table_ptr, row_major(batch_size))
        .reshape(Coord(Int64(batch_size), Int64(1)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    # Scales resolve through block 2 -- distinct from what lookup_table
    # points at, and only reachable through scales_lookup_table.
    var scales_lookup_table_ptr = unsafe_alloc[UInt32](batch_size)
    scales_lookup_table_ptr[0] = UInt32(2)
    var scales_lookup_table_opt: OptionalReg[
        _ScaledCollection.CacheType.lookup_table_tt_type
    ] = (
        TileTensor(scales_lookup_table_ptr, row_major(batch_size))
        .reshape(Coord(Int64(batch_size), Int64(1)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    var scales = _uninit_scales_pool[num_scale_blocks]().fill(
        Scalar[_ScaledCollection.scale_dtype](-1.0)
    )
    # Block 0 -- where lookup_table points -- holds a decoy value that a
    # buggy fallback to lookup_table would read instead.
    scales[0, 0, 0, 0, 0, 0] = Scalar[_ScaledCollection.scale_dtype](111.0)
    # Block 2 -- where scales_lookup_table points -- holds the real value.
    scales[2, 0, 0, 0, 0, 0] = Scalar[_ScaledCollection.scale_dtype](222.0)

    var scales_opt: OptionalReg[_ScaledCollection.scales_tt_type] = scales

    var collection = _ScaledCollection(
        rebind[_ScaledCollection.blocks_tt_type](blocks.as_unsafe_any_origin()),
        cache_lengths.as_imm().as_unsafe_any_origin(),
        lookup_table.as_imm().as_unsafe_any_origin(),
        UInt32(_rows_view_page_size),
        UInt32(_rows_view_page_size),
        scales_opt,
        scales_lookup_table_opt,
    )

    var key_cache = collection.get_key_cache(0)
    var scale = key_cache.load_scale[width=1](
        bs=0, head_idx=0, tok_idx=0, head_dim_idx=0
    )

    assert_true(
        Float32(scale) == Float32(222.0),
        String("scale resolved through the wrong LUT: got ")
        + String(scale)
        + ", expected 222.0 (via scales_lookup_table, not lookup_table)",
    )

    blocks.ptr.unsafe_free()
    lookup_table_ptr.unsafe_free()
    scales_lookup_table_ptr.unsafe_free()
    scales.ptr.unsafe_free()


def _build_rows_view_collection[
    num_rows: Int,
    value_lut_cols: Int,
    scale_lut_cols: Int,
    distinct_scale_lut: Bool,
]() raises -> _ScaledCollection:
    """Builds a scaled collection whose rows resolve distinct value and
    scale pages, so a row view can be checked against the whole collection.

    With `distinct_scale_lut` the scales get their own table and block ids;
    otherwise the constructor aliases the values table.
    """
    comptime assert (
        distinct_scale_lut or scale_lut_cols == value_lut_cols
    ), "an aliased scales table has the values table's width"
    comptime num_value_blocks = num_rows * value_lut_cols
    comptime num_scale_blocks = (
        num_value_blocks
        + num_rows
        * scale_lut_cols if distinct_scale_lut else num_rows
        * scale_lut_cols
    )
    # Distinct scale ids start past the value pool, so no scale id equals a
    # value id.
    comptime scale_id_base = num_value_blocks if distinct_scale_lut else 0

    var blocks = _zeroed_blocks_pool[num_value_blocks]()
    # Each row resolves its own value page per column, so no two rows share
    # a block id in the same column.
    for row in range(num_rows):
        for col in range(value_lut_cols):
            blocks[row * value_lut_cols + col, 0, 0, 0, 0, 0] = Float32(
                1000 + row * 10 + col
            )

    var cache_lengths_ptr = unsafe_alloc[UInt32](num_rows)
    for row in range(num_rows):
        cache_lengths_ptr[row] = UInt32(_rows_view_page_size * (row + 1))
    var cache_lengths = (
        TileTensor(cache_lengths_ptr, row_major(num_rows))
        .reshape(Coord(Int64(num_rows)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    var lookup_table_ptr = unsafe_alloc[UInt32](num_rows * value_lut_cols)
    for row in range(num_rows):
        for col in range(value_lut_cols):
            lookup_table_ptr[row * value_lut_cols + col] = UInt32(
                row * value_lut_cols + col
            )
    var lookup_table = (
        TileTensor(lookup_table_ptr, row_major(num_rows * value_lut_cols))
        .reshape(Coord(Int64(num_rows), Int64(value_lut_cols)))
        .as_imm()
        .as_unsafe_any_origin()
    )

    var scales_lookup_table_opt: OptionalReg[
        _ScaledCollection.CacheType.lookup_table_tt_type
    ] = None
    comptime if distinct_scale_lut:
        # The scales table has its own width and its own ids, distinct per
        # row and per column from the values table's.
        var scales_lookup_table_ptr = unsafe_alloc[UInt32](
            num_rows * scale_lut_cols
        )
        for row in range(num_rows):
            for col in range(scale_lut_cols):
                scales_lookup_table_ptr[row * scale_lut_cols + col] = UInt32(
                    scale_id_base + row * scale_lut_cols + col
                )
        scales_lookup_table_opt = (
            TileTensor(
                scales_lookup_table_ptr, row_major(num_rows * scale_lut_cols)
            )
            .reshape(Coord(Int64(num_rows), Int64(scale_lut_cols)))
            .as_imm()
            .as_unsafe_any_origin()
        )

    var scales = _uninit_scales_pool[num_scale_blocks]().fill(
        Scalar[_ScaledCollection.scale_dtype](0)
    )
    for row in range(num_rows):
        for col in range(scale_lut_cols):
            scales[
                scale_id_base + row * scale_lut_cols + col, 0, 0, 0, 0, 0
            ] = Scalar[_ScaledCollection.scale_dtype](2000 + row * 10 + col)

    var scales_opt: OptionalReg[_ScaledCollection.scales_tt_type] = scales

    return _ScaledCollection(
        rebind[_ScaledCollection.blocks_tt_type](blocks.as_unsafe_any_origin()),
        cache_lengths,
        lookup_table,
        UInt32(_rows_view_page_size * value_lut_cols),
        UInt32(_rows_view_page_size * value_lut_cols),
        scales_opt,
        scales_lookup_table_opt,
    )


def _free_rows_view_collection[
    distinct_scale_lut: Bool
](collection: _ScaledCollection):
    """Frees the pools and tables `_build_rows_view_collection` allocated."""
    collection.blocks.ptr.unsafe_free()
    collection.cache_lengths.ptr.unsafe_mut_cast[True]().unsafe_free()
    collection.lookup_table.ptr.unsafe_mut_cast[True]().unsafe_free()
    collection.scales.value().ptr.unsafe_free()
    # An aliased scales table is the values table, freed above.
    comptime if distinct_scale_lut:
        collection.scales_lookup_table.ptr.unsafe_mut_cast[True]().unsafe_free()


def test_paged_rows_view_resolves_the_rows_of_the_whole_collection() raises:
    """A row view resolves the same lengths, values and max lengths as the
    whole collection at the offset rows.
    """
    var whole = _build_rows_view_collection[3, 2, 2, True]()
    var S = UInt32(7)
    var C = UInt32(9)
    var view = whole.rows_view(1, 2, S, C)

    assert_true(view.max_seq_length == S, "view max_seq_length must change")
    assert_true(view.max_cache_length == C, "view max_cache_length must change")
    var view_key = view.get_key_cache(0)
    var whole_key = whole.get_key_cache(0)
    for i in range(2):
        assert_true(
            view.cache_length(i) == whole.cache_length(1 + i),
            "view cache_length must follow the offset rows",
        )
        # One token per page, so the two positions cover different pages.
        for tok in range(2):
            var view_val = view_key.block_paged_ptr[1](i, tok, 0)[0]
            var whole_val = whole_key.block_paged_ptr[1](1 + i, tok, 0)[0]
            assert_true(
                view_val == whole_val,
                "view key cache must resolve the same element as the whole",
            )

    _free_rows_view_collection[True](whole)


def test_paged_rows_view_values_and_scales_follow_their_own_tables() raises:
    """A row view resolves values and scales through their own tables'
    widths, not each other's column counts.
    """
    var whole = _build_rows_view_collection[3, 2, 3, True]()
    var view = whole.rows_view(1, 2, UInt32(7), UInt32(9))

    var view_key = view.get_key_cache(0)
    var whole_key = whole.get_key_cache(0)
    for i in range(2):
        # One token per page, so the two positions cover different columns
        # of both tables.
        for tok in range(2):
            var view_val = view_key.block_paged_ptr[1](i, tok, 0)[0]
            var whole_val = whole_key.block_paged_ptr[1](1 + i, tok, 0)[0]
            assert_true(
                view_val == whole_val,
                "view values must resolve through the values table's width",
            )
            var view_scale = view_key.load_scale[width=1](i, 0, tok, 0)
            var whole_scale = whole_key.load_scale[width=1](1 + i, 0, tok, 0)
            assert_true(
                view_scale == whole_scale,
                "view scales must resolve through the scales table's width",
            )

    _free_rows_view_collection[True](whole)


def test_paged_rows_view_scales_follow_the_aliased_values_table() raises:
    """A row view resolves scales through the values table when the scales
    table is its alias, at the values table's width.
    """
    var whole = _build_rows_view_collection[3, 2, 2, False]()
    var view = whole.rows_view(1, 2, UInt32(7), UInt32(9))

    var view_key = view.get_key_cache(0)
    var whole_key = whole.get_key_cache(0)
    for i in range(2):
        # One token per page, so the two positions cover different scale
        # columns of the values table.
        for tok in range(2):
            var view_scale = view_key.load_scale[width=1](i, 0, tok, 0)
            var whole_scale = whole_key.load_scale[width=1](1 + i, 0, tok, 0)
            assert_true(
                view_scale == whole_scale,
                "view scales must resolve through the aliased values table",
            )

    _free_rows_view_collection[False](whole)


def _build_collection_and_page_stride[
    page_size: Int, num_layers: Int
](page_stride: Int) raises -> Int:
    """Returns the byte distance between consecutive pages of a collection.

    Builds a paged collection over `num_blocks` pages, then measures the page
    stride the way a kernel experiences it: the distance between the pointers
    two adjacent physical pages resolve to. `page_stride` is forwarded
    verbatim, so -1 exercises the packed path.
    """
    comptime num_blocks = 8
    comptime batch_size = 1
    comptime shape = IndexList[6](
        num_blocks,
        2,
        num_layers,
        page_size,
        kv_params.num_heads,
        kv_params.head_size,
    )

    # The pool owes `num_blocks * page_stride` elements when a padded stride is
    # supplied, since the tensor below still describes packed pages.
    var packed_page = (
        2 * num_layers * page_size * kv_params.num_heads * kv_params.head_size
    )
    var page = packed_page if page_stride < 0 else page_stride
    var blocks_ptr = List(length=num_blocks * page, fill=Float32(0))
    comptime Collection = PagedKVCacheCollection[
        .float32,
        kv_params,
        page_size,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    var blocks = TileTensor(
        Span(blocks_ptr)[: shape.flattened_length()],
        row_major(shape.flattened_length()),
    ).reshape(
        Coord(
            Int64(shape[0]),
            Idx[2],
            Int64(shape[2]),
            Idx[page_size],
            Idx[kv_params.num_heads],
            Idx[kv_params.head_size],
        )
    )

    var cache_lengths_ptr = List(length=batch_size, fill=UInt32(0))
    var cache_lengths = TileTensor(
        Span(cache_lengths_ptr), row_major(len(cache_lengths_ptr))
    ).reshape(Coord(Int64(batch_size)))
    var lookup_table_ptr = List(length=batch_size * num_blocks, fill=UInt32(0))
    var lookup_table = TileTensor(
        Span(lookup_table_ptr), row_major(len(lookup_table_ptr))
    ).reshape(Coord(Int64(batch_size), Int64(num_blocks)))
    for j in range(num_blocks):
        lookup_table[0, j] = UInt32(j)

    var collection = Collection(
        rebind[Collection.blocks_tt_type](blocks.as_unsafe_any_origin()),
        cache_lengths.as_imm().as_unsafe_any_origin(),
        lookup_table.as_imm().as_unsafe_any_origin(),
        UInt32(page_size * num_blocks),
        UInt32(page_size * num_blocks),
        page_stride=page_stride,
    )

    var cache = collection.get_key_cache(0)
    var page0 = cache.block_paged_ptr[page_size](0, 0, 0)
    var page1 = cache.block_paged_ptr[page_size](0, page_size, 0)
    return Int(page1) - Int(page0)


def test_page_stride_of_minus_one_keeps_pages_packed() raises:
    """A -1 stride must reproduce the geometry callers already depend on."""
    comptime page_size = 16
    comptime num_layers = 3
    var measured = _build_collection_and_page_stride[page_size, num_layers](-1)
    comptime packed = (
        2 * num_layers * page_size * kv_params.num_heads * kv_params.head_size
    )
    comptime element_size = size_of[Float32]()
    assert_true(
        measured == packed * element_size,
        String("packed page stride should be ")
        + String(packed * element_size)
        + " bytes, got "
        + String(measured),
    )


def test_page_stride_overrides_the_packed_distance() raises:
    """A padded page is reached at the supplied stride, not the packed one."""
    comptime page_size = 16
    comptime num_layers = 3
    comptime packed = (
        2 * num_layers * page_size * kv_params.num_heads * kv_params.head_size
    )
    # Whole-row padding keeps the row division in `_stride()` exact.
    comptime row = kv_params.num_heads * kv_params.head_size
    comptime padded = packed + 16 * row
    comptime assert padded % row == 0, "a padded page must be whole rows"

    var measured = _build_collection_and_page_stride[page_size, num_layers](
        padded
    )
    comptime element_size = size_of[Float32]()
    assert_true(
        measured == padded * element_size,
        String("padded page stride should be ")
        + String(padded * element_size)
        + " bytes, got "
        + String(measured),
    )


def main() raises:
    test_page_stride_of_minus_one_keeps_pages_packed()
    test_page_stride_overrides_the_packed_distance()
    test_paged_kv_cache_stride_is_runtime()
    test_paged_kv_cache_offset_correctness()
    test_paged_kv_cache_quantization()
    test_scales_resolve_through_their_own_lookup_table()
    test_paged_rows_view_resolves_the_rows_of_the_whole_collection()
    test_paged_rows_view_values_and_scales_follow_their_own_tables()
    test_paged_rows_view_scales_follow_the_aliased_values_table()
    do_test[16, 16]()
    do_test[64, 16]()
    do_test[128, 64]()
    do_test[128, 64, DType.float8_e4m3fn]()
