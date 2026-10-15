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

"""Check dense rank-3/rank-4 additive masks against host attention."""

from std.math import exp, rsqrt
from std.testing import assert_almost_equal, assert_equal, assert_raises

from max.gpu.host import DeviceContext
from max.gpu.host.info import H100
from layout import Coord, MixedLayout, TileTensor, coord, row_major
from layout._host_device_tile_tensor import HostDeviceTileTensor
from nn.attention.gpu.mha import flash_attention, flash_attention_ragged
from nn.attention.mha_mask import MASK_VALUE, NullMask
from nn.attention.mha_utils import FlashAttentionAlgorithm, MHAConfig


def test_dense_materialized_mask[dtype: DType, mask_rank: Int]() raises:
    comptime batch_size = 2
    comptime seq_len = 5
    comptime num_heads = 2
    comptime kv_heads = 1
    comptime depth = 64
    comptime mask_heads = num_heads if mask_rank == 4 else 1
    comptime scale = rsqrt(Float32(depth))

    var ctx = DeviceContext()
    # SM90 register/shared WGMMA currently supports BF16, not FP16.
    comptime config = MHAConfig[dtype](
        num_heads,
        depth,
        algorithm=FlashAttentionAlgorithm(
            2 if ctx.default_device_info == H100 and dtype == .float16 else -1
        ),
    )
    var q = HostDeviceTileTensor[dtype](
        row_major[batch_size, seq_len, num_heads, depth](), ctx
    )
    var k = HostDeviceTileTensor[dtype](
        row_major[batch_size, seq_len, kv_heads, depth](), ctx
    )
    var v = HostDeviceTileTensor[dtype](
        row_major[batch_size, seq_len, kv_heads, depth](), ctx
    )
    var mask = HostDeviceTileTensor[.float32](
        row_major[batch_size, mask_heads, seq_len, seq_len](), ctx
    )
    var output = HostDeviceTileTensor[dtype](
        row_major[batch_size, seq_len, num_heads, depth](), ctx
    )
    var q_host = q.host_tensor()
    var k_host = k.host_tensor()
    var v_host = v.host_tensor()
    var mask_host = mask.host_tensor()
    comptime assert q_host.flat_rank == 4
    comptime assert k_host.flat_rank == 4
    comptime assert v_host.flat_rank == 4
    comptime assert mask_host.flat_rank == 4
    for b in range(batch_size):
        for i in range(seq_len):
            for h in range(num_heads):
                for d in range(depth):
                    q_host[b, i, h, d] = Scalar[dtype](
                        Float32(b + h + i + d % 5 - 2) * 0.03125
                    )
            for d in range(depth):
                k_host[b, i, 0, d] = Scalar[dtype](
                    Float32((b + i + 1) * (d % 3 - 1)) * 0.0625
                )
                v_host[b, i, 0, d] = Scalar[dtype](
                    Float32((b * 3 + i * 7 + d) % 17 - 8) * 0.125
                )
            for h in range(mask_heads):
                for j in range(seq_len):
                    # Every row has visible keys; rank 4 varies by head too.
                    mask_host[b, h, i, j] = (
                        Float32(MASK_VALUE) if (b + h + i + j) % 3
                        == 0 else Float32(j - i + h) * 0.25
                    )

    _ = output.host_tensor().fill(-17)
    output.to_device()
    q.to_device()
    k.to_device()
    v.to_device()
    mask.to_device()
    var q_device = q.device_tensor().as_imm()
    var k_device = k.device_tensor().as_imm()
    var v_device = v.device_tensor().as_imm()
    var output_device = output.device_tensor()
    comptime if mask_rank == 3:
        var mask_device = (
            mask.device_tensor()
            .as_imm()
            .reshape(coord[batch_size, seq_len, seq_len])
        )
        flash_attention[config=config](
            output_device,
            q_device,
            k_device,
            v_device,
            mask_device,
            scale,
            ctx,
        )
    else:
        flash_attention[config=config](
            output_device,
            q_device,
            k_device,
            v_device,
            mask.device_tensor().as_imm(),
            scale,
            ctx,
        )
    output.to_host()
    var output_host = output.host_tensor()
    comptime assert output_host.flat_rank == 4

    # Independent stable QK-softmax-PV reference, using stored input values.
    for b in range(batch_size):
        for i in range(seq_len):
            for h in range(num_heads):
                var scores = List[Float32](length=seq_len, fill=0)
                var row_max = Float32(MASK_VALUE)
                for j in range(seq_len):
                    var dot = Float32(0)
                    for d in range(depth):
                        dot += Float32(q_host[b, i, h, d]) * Float32(
                            k_host[b, j, h // (num_heads // kv_heads), d]
                        )
                    var mask_head = h if mask_rank == 4 else 0
                    scores[j] = dot * scale + mask_host[b, mask_head, i, j]
                    row_max = max(row_max, scores[j])
                var denom = Float32(0)
                for j in range(seq_len):
                    scores[j] = exp(scores[j] - row_max)
                    denom += scores[j]
                for d in range(depth):
                    var expected = Float32(0)
                    for j in range(seq_len):
                        expected += scores[j] * Float32(
                            v_host[b, j, h // (num_heads // kv_heads), d]
                        )
                    assert_almost_equal(
                        Float32(output_host[b, i, h, d]),
                        expected / denom,
                        atol=2e-3,
                        rtol=2e-2,
                    )


def test_ragged_rejects_strided_offsets() raises:
    var ctx = DeviceContext()
    comptime config = MHAConfig[.float16](
        1,
        64,
        algorithm=FlashAttentionAlgorithm(
            2 if ctx.default_device_info == H100 else -1
        ),
    )
    var qkv = HostDeviceTileTensor[.float16](row_major[1, 1, 64](), ctx)
    var output = HostDeviceTileTensor[.float16](row_major[1, 1, 64](), ctx)
    var offsets = HostDeviceTileTensor[.uint32](row_major[4](), ctx)
    var max_prompt_len = HostDeviceTileTensor[.uint32](row_major[1]())
    _ = qkv.host_tensor().fill(0)
    _ = output.host_tensor().fill(-17)
    var offsets_host = offsets.host_tensor().fill(1)
    comptime assert offsets_host.flat_rank == 1
    offsets_host[0] = 0
    _ = max_prompt_len.host_tensor().fill(1)
    qkv.to_device()
    output.to_device()
    offsets.to_device()

    # Dynamic stride must be rejected before the contiguous metadata ABI is built.
    var stride = 2
    var strided_offsets = TileTensor(
        offsets.device_tensor().as_imm().unsafe_ptr(),
        MixedLayout(coord[2], Coord(stride)),
    )
    with assert_raises(contains="input_row_offsets must have unit stride"):
        flash_attention_ragged[config=config](
            output.device_tensor(),
            qkv.device_tensor().as_imm(),
            qkv.device_tensor().as_imm(),
            qkv.device_tensor().as_imm(),
            strided_offsets,
            max_prompt_len.host_tensor().as_imm(),
            NullMask(),
            Float32(0.125),
            ctx,
        )
    output.to_host()
    var output_host = output.host_tensor()
    comptime assert output_host.flat_rank == 3
    for d in range(64):
        assert_equal(output_host[0, 0, d], Float16(-17))


def main() raises:
    test_dense_materialized_mask[.float16, 3]()
    test_dense_materialized_mask[.float16, 4]()
    test_dense_materialized_mask[.bfloat16, 3]()
    test_dense_materialized_mask[.bfloat16, 4]()
    test_ragged_rejects_strided_offsets()
