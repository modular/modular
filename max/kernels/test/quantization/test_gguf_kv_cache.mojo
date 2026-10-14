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

from std.math import inf
from std.testing import assert_almost_equal, assert_equal

from max.gpu.host import DeviceContext
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, TileTensor, row_major
from nn.kv_cache_ragged import unfused_qkv_matmul_ragged_paged_gguf_quantized
from test_qmatmul_k import (
    GemmContext,
    QuantizedGemm,
    qgemm_Q4_0,
    qgemm_Q4_K,
    qgemm_Q6_K,
    reference_gemm,
)


def check_paged_qkv[
    qgemm: QuantizedGemm, encoding: StaticString
](ctx: DeviceContext) raises:
    comptime M = 3
    comptime N = 64
    comptime K = 256
    comptime PAGE = 16
    comptime params = KVCacheStaticParams(num_heads=1, head_size=N)
    var q = GemmContext[qgemm](M, N, K)
    var k = GemmContext[qgemm](M, N, K)
    var v = GemmContext[qgemm](M, N, K)
    reference_gemm[qgemm](q.a, q.b, q.c_golden)
    reference_gemm[qgemm](q.a, k.b, k.c_golden)
    reference_gemm[qgemm](q.a, v.b, v.c_golden)

    var output_storage = List(length=(M + 1) * N, fill=inf[.float32]())
    var output = TileTensor(Span(output_storage), row_major((M, Idx[N])))
    var block_storage = List(length=5 * 2 * 2 * PAGE * N, fill=inf[.float32]())
    var blocks = TileTensor(
        Span(block_storage),
        row_major((Idx[5], Idx[2], Int(2), Idx[PAGE], Idx[1], Idx[N])),
    )
    var offsets_storage: List[UInt32] = [0, 2, 3]
    var offsets = TileTensor(Span(offsets_storage), row_major(Int(3)))
    var lengths_storage: List[UInt32] = [15, 3]
    var lengths = TileTensor(Span(lengths_storage), row_major(Int(2)))
    var lookup_storage: List[UInt32] = [3, 1, 2, 0]
    var lookup = TileTensor(Span(lookup_storage), row_major((Int(2), Int(2))))
    var cache = PagedKVCacheCollection[
        .float32, params, PAGE, scales_origin=MutUntrackedOrigin
    ](blocks, lengths.as_imm(), lookup.as_imm(), UInt32(2), UInt32(17))

    var hidden = q.a.reshape(row_major((Int(M), Idx[K]))).as_imm()
    var qw = q.b_packed.reshape(
        row_major((Idx[N], Int(q.b_packed.dim[1]())))
    ).as_imm()
    var kw = k.b_packed.reshape(
        row_major((Idx[N], Int(k.b_packed.dim[1]())))
    ).as_imm()
    var vw = v.b_packed.reshape(
        row_major((Idx[N], Int(v.b_packed.dim[1]())))
    ).as_imm()
    unfused_qkv_matmul_ragged_paged_gguf_quantized[
        quantization_encoding_q=encoding,
        quantization_encoding_k=encoding,
        quantization_encoding_v=encoding,
    ](
        hidden,
        offsets.as_imm(),
        qw,
        kw,
        vw,
        cache,
        UInt32(1),
        output,
        ctx,
    )

    for row in range(M):
        for col in range(N):
            assert_almost_equal(
                output[row, col], q.c_golden[row, col], atol=1e-4, rtol=1e-4
            )
    for i in range(M * N, len(output_storage)):
        assert_equal(output_storage[i], inf[.float32]())

    # Check every physical cache slot, including prefixes and the other layer.
    for page in range(5):
        for kv in range(2):
            for layer in range(2):
                for token in range(PAGE):
                    var row = -1
                    if layer == 1:
                        if page == 3 and token == 15:
                            row = 0
                        elif page == 1 and token == 0:
                            row = 1
                        elif page == 2 and token == 3:
                            row = 2
                    for col in range(N):
                        var actual = blocks[page, kv, layer, token, 0, col]
                        if row >= 0:
                            var expected = k.c_golden[
                                row, col
                            ] if kv == 0 else (v.c_golden[row, col])
                            assert_almost_equal(
                                actual, expected, atol=1e-4, rtol=1e-4
                            )
                        else:
                            assert_equal(actual, inf[.float32]())
    q.free()
    k.free()
    v.free()


def main() raises:
    with DeviceContext(api="cpu") as ctx:
        check_paged_qkv[qgemm_Q4_0, "q4_0"](ctx)
        check_paged_qkv[qgemm_Q4_K, "q4_k"](ctx)
        check_paged_qkv[qgemm_Q6_K, "q6_k"](ctx)
