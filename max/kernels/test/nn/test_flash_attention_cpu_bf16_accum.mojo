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
"""Checks that CPU flash attention on bf16 inputs matches its fp32 path.

Both runs see the same bf16-representable values, so any gap beyond the final
bf16 rounding of the output comes from accumulating in bf16.
"""

from std.math import isclose, rsqrt
from std.random import randn_float64, seed

from kv_cache.types import (
    ContinuousBatchingKVCacheCollection,
    KVCacheStaticParams,
)
from layout import Coord, Idx, TileTensor, row_major
from nn.attention.cpu.mha import flash_attention_kv_cache
from nn.attention.mha_mask import CausalMask
from std.testing import assert_equal

comptime kv_params = KVCacheStaticParams(num_heads=2, head_size=128)
comptime num_q_heads = 4
# Long enough that a bf16 running sum over the keys drops the small terms.
comptime context_len = 4096
comptime num_blocks = 1
comptime num_layers = 1


def run_attention[
    dtype: DType
](
    q_values: List[Float32], kv_values: List[Float32], seq_len: Int
) raises -> List[Float32]:
    comptime BlocksLayout = ContinuousBatchingKVCacheCollection[
        dtype,
        kv_params,
        MutUntrackedOrigin,
        ImmUntrackedOrigin,
        ImmUntrackedOrigin,
    ].blocks_tt_layout

    var cache_len = context_len - seq_len

    var q_buf = List(
        length=len(q_values),
        fill_with=lambda (i: Int) {imm} -> Scalar[dtype]: q_values[i].cast[
            dtype
        ](),
    )
    var q = TileTensor(Span(q_buf), row_major(len(q_buf))).reshape(
        Coord(
            Int64(1),
            Int64(seq_len),
            Idx[num_q_heads],
            Idx[kv_params.head_size],
        )
    )

    var output_buf = List(length=len(q_values), fill=Scalar[dtype](0))
    var output = TileTensor(
        Span(output_buf), row_major(len(output_buf))
    ).reshape(
        Coord(
            Int64(1),
            Int64(seq_len),
            Idx[num_q_heads],
            Idx[kv_params.head_size],
        )
    )

    var kv_block_buf = List(
        length=len(kv_values),
        fill_with=lambda (i: Int) {imm} -> Scalar[dtype]: kv_values[i].cast[
            dtype
        ](),
    )
    var blocks_shape = Coord[*BlocksLayout.shape_types]()
    blocks_shape[0] = Int64(num_blocks)
    blocks_shape[1] = Int64(2)
    blocks_shape[2] = Int64(num_layers)
    blocks_shape[3] = Int64(context_len)
    var blocks_strides = Coord[*BlocksLayout.stride_types]()
    blocks_strides[2] = blocks_shape[3] * Int64(blocks_strides[3].value())
    blocks_strides[1] = blocks_shape[2] * blocks_strides[2]
    blocks_strides[0] = blocks_shape[1] * blocks_strides[1]
    var kv_block = TileTensor(
        Span(kv_block_buf), row_major(len(kv_block_buf))
    ).reshape(BlocksLayout(blocks_shape, blocks_strides))

    var cache_lengths_buf: List[UInt32] = [UInt32(cache_len)]
    var cache_lengths = TileTensor(
        Span(cache_lengths_buf), row_major(1)
    ).reshape(Coord(Int64(1)))
    var lookup_table_buf: List[UInt32] = [UInt32(0)]
    var lookup_table = TileTensor(Span(lookup_table_buf), row_major(1)).reshape(
        Coord(Int64(1))
    )

    var kv_collection = ContinuousBatchingKVCacheCollection[dtype, kv_params](
        kv_block,
        cache_lengths.as_imm(),
        lookup_table.as_imm(),
        UInt32(seq_len),
        UInt32(context_len),
    )

    flash_attention_kv_cache(
        q.as_imm(),
        kv_collection.get_key_cache(0),
        kv_collection.get_value_cache(0),
        CausalMask(),
        rsqrt(Float32(kv_params.head_size)),
        output,
    )

    return List(
        length=len(output_buf),
        fill_with=lambda (i: Int) {imm} -> Float32: output_buf[i].cast[
            DType.float32
        ](),
    )


def bf16_representable(count: Int, stddev: Float64) -> List[Float32]:
    return List(
        length=count,
        fill_with=lambda (i: Int) {imm} -> Float32: Float32(
            randn_float64(0, stddev)
        )
        .cast[DType.bfloat16]()
        .cast[DType.float32](),
    )


def check_bf16_matches_fp32(seq_len: Int) raises:
    seed(42)
    var q_values = bf16_representable(
        seq_len * num_q_heads * kv_params.head_size, 1.0
    )
    var kv_values = bf16_representable(
        num_blocks
        * 2
        * num_layers
        * context_len
        * kv_params.num_heads
        * kv_params.head_size,
        1.0,
    )

    var expected = run_attention[DType.float32](q_values, kv_values, seq_len)
    var actual = run_attention[DType.bfloat16](q_values, kv_values, seq_len)

    var max_abs_err = Float32(0)
    var mismatches = 0
    for i in range(len(expected)):
        max_abs_err = max(max_abs_err, abs(actual[i] - expected[i]))
        # About one bf16 ulp: the output is rounded to bf16 once.
        if not isclose(actual[i], expected[i], atol=5e-4, rtol=4e-3):
            mismatches += 1
    print(
        t"seq_len={seq_len} context_len={context_len}:"
        t" max abs err {max_abs_err}, mismatches {mismatches}/{len(expected)}"
    )
    assert_equal(mismatches, 0, "bf16 CPU attention drifts from fp32")


def main() raises:
    # seq_len=1 takes the GEMV decode path; 16 takes the packed GEMM path.
    check_bf16_matches_fp32(1)
    check_bf16_matches_fp32(16)
