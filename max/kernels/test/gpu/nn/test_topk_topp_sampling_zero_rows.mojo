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
"""Tests that top-k/top-p sampling over zero rows launches nothing.

A speculative-decoding step that skips its drafter samples zero draft rows.
`topk_topp_sampling_from_prob` dispatches to two launchers: the SM90+ cluster
kernel that emits the distribution from logits, and the single-block kernel
for everything else. Both must return instead of launching an empty grid,
which `enqueue_function` rejects.
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from std.testing import assert_equal

from nn.sampling import topk_topp_sampling_from_prob

comptime _D = 4096
comptime _TOKEN_SENTINEL = Int64(-7)
comptime _DIST_SENTINEL = Float32(-3.0)


def run_zero_rows[
    from_logits: Bool, emit_dist: Bool
](ctx: DeviceContext) raises:
    # The buffers hold one row so the allocation is non-empty, but the
    # tensors handed to the launcher view none of it.
    var logits_host = ctx.enqueue_create_host_buffer[.float32](_D)
    for col in range(_D):
        logits_host[col] = Float32(col % 17) / 17.0
    var logits_dev = ctx.enqueue_create_buffer[.float32](_D)
    ctx.enqueue_copy(logits_dev, logits_host)

    var tokens_host = ctx.enqueue_create_host_buffer[.int64](1)
    tokens_host[0] = _TOKEN_SENTINEL
    var tokens_dev = ctx.enqueue_create_buffer[.int64](1)
    ctx.enqueue_copy(tokens_dev, tokens_host)

    var dist_host = ctx.enqueue_create_host_buffer[.float32](_D)
    for col in range(_D):
        dist_host[col] = _DIST_SENTINEL
    var dist_dev = ctx.enqueue_create_buffer[.float32](_D)
    ctx.enqueue_copy(dist_dev, dist_host)

    var seed_dev = ctx.enqueue_create_buffer[.uint64](1)
    var top_k_dev = ctx.enqueue_create_buffer[.int64](1)
    var top_p_dev = ctx.enqueue_create_buffer[.float32](1)
    var temperature_dev = ctx.enqueue_create_buffer[.float32](1)

    var out_dist = Optional(
        TileTensor(dist_dev, row_major(0, _D)).as_unsafe_any_origin()
    )
    comptime if not emit_dist:
        out_dist = None

    topk_topp_sampling_from_prob[
        from_logits=from_logits, emit_dist=emit_dist, dist_dtype=DType.float32
    ](
        ctx,
        TileTensor(logits_dev, row_major(0, _D)),
        TileTensor(tokens_dev, row_major(0)),
        50,
        rng_seed=TileTensor(seed_dev, row_major(0))
        .as_unsafe_any_origin()
        .as_imm(),
        top_k_arr=TileTensor(top_k_dev, row_major(0))
        .as_unsafe_any_origin()
        .as_imm(),
        top_p_arr=TileTensor(top_p_dev, row_major(0))
        .as_unsafe_any_origin()
        .as_imm(),
        temperature=TileTensor(temperature_dev, row_major(0))
        .as_unsafe_any_origin()
        .as_imm(),
        out_dist=out_dist,
    )

    ctx.enqueue_copy(tokens_host, tokens_dev)
    ctx.enqueue_copy(dist_host, dist_dev)
    ctx.synchronize()

    var label = String(t"from_logits={from_logits} emit_dist={emit_dist}")
    assert_equal(
        tokens_host[0],
        _TOKEN_SENTINEL,
        msg=String(t"{label}: a zero-row call wrote a token"),
    )
    for col in range(_D):
        assert_equal(
            dist_host[col],
            _DIST_SENTINEL,
            msg=String(t"{label} col={col}: a zero-row call wrote the dist"),
        )

    _ = logits_dev^
    _ = tokens_dev^
    _ = dist_dev^
    _ = seed_dev^
    _ = top_k_dev^
    _ = top_p_dev^
    _ = temperature_dev^


def main() raises:
    with DeviceContext() as ctx:
        # The cluster launcher on SM90+, the single-block one elsewhere.
        run_zero_rows[from_logits=True, emit_dist=True](ctx)
        # The single-block launcher on every device.
        run_zero_rows[from_logits=True, emit_dist=False](ctx)
        run_zero_rows[from_logits=False, emit_dist=False](ctx)
