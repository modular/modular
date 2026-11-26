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
"""SM100 FA4 cross-stage P, last KV block: P1(n) must not overwrite P1(n-1).

In the 2Q cross-stage-P schedule softmax warpgroup 1 stores P1(n) into the S0
columns that the MMA warp reads P1(n-1) from for PV1(n-1). Before the last KV
block, the peer token orders the two stores; the last block has no QK0, so
softmax warpgroup 1 has to wait for PV1(n-1) itself. If it does not, a late
MMA warp either reads P1(n) in place of P1(n-1) (wrong output) or waits for an
S1 consumer-barrier phase that warpgroup 1 already lapped (hang).

The BUILD variant of this test sets `FA4_TEST_DELAY_LAST_PV1_NS`, which stalls
the MMA warp before that PV1 and makes the race deterministic. B200 only.
"""

from std.random import rand, seed

from max.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.mha import flash_attention, mha_gpu_naive
from nn.attention.mha_mask import NullMask
from std.testing import assert_equal


def test[
    num_heads: Int,
](seq_len: Int, num_keys: Int, ctx: DeviceContext[]) raises:
    print(
        "test_mha_sm100_crossp_last_block heads:",
        num_heads,
        "seq:",
        seq_len,
        "keys:",
        num_keys,
    )
    comptime depth = 128
    comptime qkv_type = DType.bfloat16
    comptime scale = Float32(0.125)

    var size = seq_len * num_heads * depth
    var kv_size = num_keys * num_heads * depth
    var q_host = ctx.enqueue_create_host_buffer[qkv_type](size)
    var k_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var v_host = ctx.enqueue_create_host_buffer[qkv_type](kv_size)
    var out_host = ctx.enqueue_create_host_buffer[qkv_type](size)
    var ref_host = ctx.enqueue_create_host_buffer[qkv_type](size)
    seed(1234567890)
    rand[qkv_type](q_host.as_span())
    rand[qkv_type](k_host.as_span())
    rand[qkv_type](v_host.as_span())

    var q_dev = ctx.enqueue_create_buffer[qkv_type](size)
    var k_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var v_dev = ctx.enqueue_create_buffer[qkv_type](kv_size)
    var out_dev = ctx.enqueue_create_buffer[qkv_type](size)
    var ref_dev = ctx.enqueue_create_buffer[qkv_type](size)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(v_dev, v_host)

    var q_shape = row_major((1, seq_len, Idx[num_heads], Idx[depth]))
    var kv_shape = row_major((1, num_keys, Idx[num_heads], Idx[depth]))
    var q = TileTensor(q_dev, q_shape)
    var k = TileTensor(k_dev, kv_shape)
    var v = TileTensor(v_dev, kv_shape)
    var out = TileTensor(out_dev, q_shape)
    var out_ref = TileTensor(ref_dev, q_shape)

    flash_attention(out, q, k, v, NullMask(), scale, ctx, None)
    mha_gpu_naive(
        q,
        k,
        v,
        NullMask(),
        out_ref,
        scale,
        1,
        seq_len,
        num_keys,
        num_heads,
        depth,
        1,
        ctx,
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.synchronize()

    # FNV-1a over the output bits, so two builds can be compared bit for bit.
    var digest = UInt64(0xCBF29CE484222325)
    for i in range(size):
        var bits = UInt64(out_host[i].cast[DType.float32]().to_bits[.uint32]())
        digest = (digest ^ bits) * UInt64(0x100000001B3)
    print("output digest:", digest)

    var bad = 0
    var first_bad = -1
    for i in range(size):
        var got = out_host[i].cast[DType.float64]()
        var want = ref_host[i].cast[DType.float64]()
        if abs(got - want) > 1e-5 + 1e-2 * abs(want):
            bad += 1
            if first_bad < 0:
                first_bad = i
    assert_equal(bad, 0, msg=String(t"first mismatch at element {first_bad}"))

    _ = q_dev^
    _ = k_dev^
    _ = v_dev^
    _ = out_dev^
    _ = ref_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Two KV blocks: the last block is also the first one with a PV1.
        test[num_heads=24](1024, 256, ctx)
        test[num_heads=24](1024, 1024, ctx)
        # The FLUX.3 Klein joint-block shape, with a ragged last Q tile.
        test[num_heads=24](4416, 4416, ctx)
