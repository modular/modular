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

from extensibility import (
    ManagedTensorSlice,
    IOSpec,
    get_row_major_tensor_spec_static,
)
from extensibility.managed_tensor_slice import (
    StaticTensorSpec,
    _IndexListToTileLayout,
)
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import RowMajorLayout, TileTensor, row_major
from nn.kv_cache import (
    generic_get_paged_cache,
    generic_get_paged_cache_with_scales,
)
from std.testing import assert_equal
from std.utils.coord import DynamicCoord
from std.utils.index import IndexList


def _graph_input[
    dtype: DType, rank: Int, *dims: Int
](
    mut data: List[Scalar[dtype]],
) -> ManagedTensorSlice[
    io_spec=IOSpec.Input,
    static_spec=get_row_major_tensor_spec_static[dtype, rank, *dims](),
]:
    comptime assert rank == len(dims)
    var shape = IndexList[rank]()
    comptime for i in range(rank):
        shape[i] = dims[i]
    return {data.unsafe_ptr(), shape}


def _view[
    dtype: DType, rank: Int
](
    ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin], shape: IndexList[rank]
) -> TileTensor[
    dtype,
    RowMajorLayout[*DynamicCoord[.int64, rank].element_types],
    MutAnyOrigin,
]:
    var coord = DynamicCoord[.int64, rank]()
    comptime for i in range(rank):
        coord[i] = rebind[coord.element_types[i]](Int64(shape[i]))
    return TileTensor(ptr, row_major(coord))


def _check[
    is_mla: Bool, padded: Bool, scaled: Bool, separate_lut: Bool
](
    collection: PagedKVCacheCollection[DType.float32, ...],
    values: List[Float32],
    scales: List[Float32],
) raises:
    comptime kv_dim = 1 if is_mla else 2
    comptime value_page = kv_dim * 2 * 2 * 4 + (8 if padded else 0)
    comptime scale_page = kv_dim * 2 * 2 + (4 if padded else 0)
    assert_equal(collection.max_seq_length, UInt32(3))
    assert_equal(collection.max_cache_length, UInt32(4))
    var expected = List(length=len(values), fill=Float32(-77))
    for layer in range(2):
        var key = collection.get_key_cache(layer)
        assert_equal(key.cache_length(0), 1)
        for tok in range(4):
            var physical_page = 2 if tok < 2 else 0
            for d in range(4):
                var value = Float32(1 + layer * 100 + tok * 4 + d)
                key.block_paged_ptr[1](0, tok, 0, d)[] = value
                var offset = (
                    physical_page * value_page + layer * 8 + (tok % 2) * 4 + d
                )
                expected[offset] = value
                comptime if not is_mla:
                    var val_cache = collection.get_value_cache(layer)
                    val_cache.block_paged_ptr[1](0, tok, 0, d)[] = value + 20
                    expected[offset + 16] = value + 20
            comptime if scaled:
                var scale_block = (
                    1 if tok < 2 else 3
                ) if separate_lut else physical_page
                var scale_offset = (
                    scale_block * scale_page + layer * 2 + tok % 2
                )
                assert_equal(
                    Float32(key.load_scale[1](0, 0, tok, 0)),
                    scales[scale_offset],
                )
                comptime if not is_mla:
                    var val_cache = collection.get_value_cache(layer)
                    assert_equal(
                        Float32(val_cache.load_scale[1](0, 0, tok, 0)),
                        scales[scale_offset + 4],
                    )
    # Checks every K/V region, unused page, and physical page padding.
    for i in range(len(values)):
        assert_equal(values[i], expected[i])


def _run[
    is_mla: Bool,
    padded: Bool,
    scaled: Bool,
    separate_lut: Bool,
    graph_wrapper: Bool = False,
]() raises:
    comptime params = KVCacheStaticParams(
        num_heads=1, head_size=4, is_mla=is_mla
    )
    comptime kv_dim = 1 if is_mla else 2
    comptime value_page = kv_dim * 16 + (8 if padded else 0)
    comptime scale_page = kv_dim * 4 + (4 if padded else 0)
    var values = List(length=4 * value_page, fill=Float32(-77))
    var scales = List(length=4 * scale_page, fill=Float32(-55))
    for page in range(4):
        for kv in range(kv_dim):
            for layer in range(2):
                for tok in range(2):
                    scales[
                        page * scale_page + kv * 4 + layer * 2 + tok
                    ] = Float32(
                        1000 + page * 100 + kv * 20 + layer * 10 + tok * 2
                    )
    var lens: List[UInt32] = [1]
    var lut: List[UInt32] = [2, 0]
    var scale_lut: List[UInt32] = [1, 3]
    var prompt: List[UInt32] = [3]
    var maximum: List[UInt32] = [4]
    var page_stride: List[Int64] = [Int64(value_page if padded else -1)]
    var scales_page_stride: List[Int64] = [Int64(scale_page if padded else -1)]
    var blocks = _view[.float32, 6](
        values.unsafe_ptr().as_unsafe_any_origin(),
        IndexList[6](4, kv_dim, 2, 2, 1, 4),
    )
    var lengths = _view[.uint32, 1](
        lens.unsafe_ptr().as_unsafe_any_origin(), IndexList[1](1)
    ).as_imm()
    var table = _view[.uint32, 2](
        lut.unsafe_ptr().as_unsafe_any_origin(), IndexList[2](1, 2)
    ).as_imm()
    var max_prompt = _view[.uint32, 1](
        prompt.unsafe_ptr().as_unsafe_any_origin(), IndexList[1](1)
    ).as_imm()
    var max_cache = _view[.uint32, 1](
        maximum.unsafe_ptr().as_unsafe_any_origin(), IndexList[1](1)
    ).as_imm()
    var value_stride = _view[.int64, 1](
        page_stride.unsafe_ptr().as_unsafe_any_origin(), IndexList[1](1)
    ).as_imm()
    comptime if scaled:
        var scale_blocks = _view[.float32, 6](
            scales.unsafe_ptr().as_unsafe_any_origin(),
            IndexList[6](4, kv_dim, 2, 2, 1, 1),
        )
        var scale_stride = _view[.int64, 1](
            scales_page_stride.unsafe_ptr().as_unsafe_any_origin(),
            IndexList[1](1),
        ).as_imm()
        var scale_table = _view[.uint32, 2](
            scale_lut.unsafe_ptr().as_unsafe_any_origin(), IndexList[2](1, 2)
        ).as_imm()
        # The scales resolve their pages through `table` itself while they
        # share the values' block-id space, and through their own LUT when
        # paged independently.
        var scales_table = scale_table if separate_lut else table
        var collection = generic_get_paged_cache_with_scales[
            .float32, .float32, params, 2, 4
        ](
            blocks,
            value_stride,
            lengths,
            table,
            max_prompt,
            max_cache,
            scale_blocks,
            scale_stride,
            scales_table,
        )
        _check[is_mla, padded, scaled, separate_lut](collection, values, scales)
    elif graph_wrapper:
        # Graph cache views historically use packed logical shapes even when
        # runtime metadata carries a different physical page pitch.
        comptime block_spec = StaticTensorSpec[
            .float32,
            6,
            static_layout=_IndexListToTileLayout[
                IndexList[6](4, kv_dim, 2, 2, 1, 4),
                IndexList[6](-1, -1, -1, -1, -1, -1),
            ],
        ](4, .GENERIC)
        var graph_blocks = ManagedTensorSlice[
            io_spec=IOSpec.MutableInput, static_spec=block_spec
        ](
            values.unsafe_ptr(),
            IndexList[6](4, kv_dim, 2, 2, 1, 4),
            IndexList[6](value_page, 16, 8, 4, 4, 1),
        )
        var graph_stride = _graph_input[.int64, 1, 1](page_stride)
        var graph_lengths = _graph_input[.uint32, 1, 1](lens)
        var graph_lookup = _graph_input[.uint32, 2, 1, 2](lut)
        var graph_prompt = _graph_input[.uint32, 1, 1](prompt)
        var graph_maximum = _graph_input[.uint32, 1, 1](maximum)
        var collection = generic_get_paged_cache(
            graph_blocks,
            graph_stride,
            graph_lengths,
            graph_lookup,
            graph_prompt,
            graph_maximum,
        )
        _check[is_mla, padded, scaled, separate_lut](collection, values, scales)
    else:
        var collection = generic_get_paged_cache[.float32, params, 2](
            blocks,
            value_stride,
            lengths,
            table,
            max_prompt,
            max_cache,
        )
        _check[is_mla, padded, scaled, separate_lut](collection, values, scales)

    # Any-origin cache metadata views must not outlive their owning Lists.
    _ = lens
    _ = lut
    _ = scale_lut
    _ = prompt
    _ = maximum
    _ = page_stride
    _ = scales_page_stride


def main() raises:
    comptime for mla in range(2):
        comptime for padded in range(2):
            _run[Bool(mla), Bool(padded), False, False]()
            _run[Bool(mla), Bool(padded), False, False, True]()
            _run[Bool(mla), Bool(padded), True, False]()
            _run[Bool(mla), Bool(padded), True, True]()
