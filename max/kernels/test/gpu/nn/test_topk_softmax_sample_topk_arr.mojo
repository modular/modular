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
"""`topk_softmax_sample`: the per-row `top_k_arr` contract.

Three invariants, one per axis of the contract:

* A row's `top_k_arr[i]` is honored exactly while it is at or below `_MAX_TOPK`,
  including when it exceeds the scalar `top_k_val`.
* A row above `_MAX_TOPK` is clamped to it. The cache is sized for the bound and
  the host cannot read this device tensor, so clamping is the only place the
  bound can be enforced.
* A `top_k_val` above `_MAX_TOPK` is rejected on the host, which *can* check it.

Logits are strictly increasing, so the true top-k of a row of `D` is exactly the
index range `[D - k, D)`.

**Each case is checked from both sides, because an upper bound alone proves
nothing here.** Every smaller top-k set sits inside every larger one, so "each
draw landed inside the true top-k" would also hold for an implementation that
ignored `top_k_arr` and sampled the scalar's top-k. Cases whose `k` is well
above the scalar therefore also require most draws to land *outside* the
scalar's top-k — an observation only an implementation honoring the per-row
value can produce.

Membership, not equality: `Atomic.fetch_add` decides the order elements are
cached in, so the sampled index may differ between runs at one seed.

Regression coverage for KERN-3691.
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from nn.sampling import topk_softmax_sample
from nn.sampling.topk_fi import _MAX_TOPK
from std.testing import assert_raises, assert_true

comptime DTYPE = DType.float32
comptime OUT_IDX = DType.int32
comptime D = 8192
comptime TRIALS = 20

# The scalar `top_k_val` every case passes. A draw below index `D - SCALAR_K` is
# outside what the scalar alone could have produced.
comptime SCALAR_K = 32

# How many of `TRIALS` draws must land outside the scalar's top-k for a case
# that asks for a much larger `k`. Weights fall off as exp(-0.001 * rank) here,
# so the scalar's 32 elements hold ~3.6% of the mass once k reaches 2048: the
# true rate is ~96%, and 40% leaves room for the pivot search landing a element
# either side of the boundary without ever making this flaky.
comptime MIN_OUTSIDE = (TRIALS * 2) // 5


def check_row(
    ctx: DeviceContext,
    label: StaticString,
    top_k_val: Int,
    per_row_k: Int,
    expect_effective_k: Int,
    min_outside_scalar: Int = 0,
) raises:
    """Draws `TRIALS` samples and checks the effective k from both sides.

    Args:
        ctx: Device context.
        label: Case name, used in assertion messages.
        top_k_val: The scalar passed to the launcher.
        per_row_k: The value written into `top_k_arr[0]`.
        expect_effective_k: The k the kernel should actually use — it differs
            from `per_row_k` when the clamp applies. No draw may fall outside
            the true top-`expect_effective_k`.
        min_outside_scalar: How many draws must land *outside* the scalar's
            top-`SCALAR_K`. Zero disables the check, for cases where the per-row
            value is too close to the scalar to separate them by sampling.
    """
    var logits_host = ctx.enqueue_create_host_buffer[DTYPE](D)
    for c in range(D):
        logits_host[c] = Float32(c) * 0.001
    var logits_dev = ctx.enqueue_create_buffer[DTYPE](D)
    ctx.enqueue_copy(logits_dev, logits_host)

    var out_dev = ctx.enqueue_create_buffer[OUT_IDX](1)
    var k_host = ctx.enqueue_create_host_buffer[OUT_IDX](1)
    k_host[0] = Scalar[OUT_IDX](per_row_k)
    var k_dev = ctx.enqueue_create_buffer[OUT_IDX](1)
    ctx.enqueue_copy(k_dev, k_host)

    var lo = D - expect_effective_k
    var scalar_lo = D - SCALAR_K
    var outside_scalar = 0

    for t in range(TRIALS):
        topk_softmax_sample[DTYPE, OUT_IDX](
            ctx,
            TileTensor(logits_dev, row_major(1, D)),
            TileTensor(out_dev, row_major(1)),
            top_k_val,
            temperature_val=1.0,
            seed_val=UInt64(t),
            top_k_arr=TileTensor(
                ptr=k_dev.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
                layout=row_major(Int64(1)),
            ),
        )
        ctx.synchronize()
        with out_dev.map_to_host() as out_host:
            var idx = Int(out_host[0])
            # Upper side: the effective k is no larger than expected.
            assert_true(
                idx >= lo and idx < D,
                String(
                    label,
                    ": sampled index ",
                    idx,
                    " is outside the true top-",
                    expect_effective_k,
                    " range [",
                    lo,
                    ", ",
                    D,
                    ")",
                ),
            )
            if idx < scalar_lo:
                outside_scalar += 1

    # Lower side: the effective k is no smaller than expected. Without this an
    # implementation that ignored `top_k_arr` would satisfy every assertion
    # above, since the scalar's top-k sits inside every larger top-k.
    assert_true(
        outside_scalar >= min_outside_scalar,
        String(
            label,
            ": only ",
            outside_scalar,
            " of ",
            TRIALS,
            " draws landed outside the scalar's top-",
            SCALAR_K,
            " (need at least ",
            min_outside_scalar,
            "). The per-row top_k_arr value does not appear to be honored.",
        ),
    )

    _ = logits_dev^
    _ = out_dev^
    _ = k_dev^


def main() raises:
    with DeviceContext() as ctx:
        # Baseline: the per-row value agrees with the scalar, so every draw
        # belongs to the scalar's top-k and none may fall outside it.
        check_row(ctx, "k == top_k_val", 32, 32, 32)

        # The boundary: one element beyond the scalar. It carries no
        # outside-the-scalar requirement, because that one element holds about
        # 1/33 of the mass — too little to assert on in a bounded number of
        # draws. The cases below carry that burden instead.
        check_row(ctx, "k == top_k_val + 1", 32, 33, 33)

        # Well above the scalar and inside the bound: honored exactly, so most
        # draws must come from outside the scalar's top-k.
        check_row(ctx, "k >> top_k_val", 32, 2048, 2048, MIN_OUTSIDE)

        # Exactly the bound — the largest k the cache is sized for.
        check_row(ctx, "k == _MAX_TOPK", 32, _MAX_TOPK, _MAX_TOPK, MIN_OUTSIDE)

        # Past the bound: clamped to it. The upper check pins the clamp to
        # `_MAX_TOPK` rather than the full row; the outside check pins it above
        # the scalar, so neither "ignore the row" nor "clamp to the scalar"
        # survives.
        check_row(
            ctx, "k > _MAX_TOPK", 32, _MAX_TOPK * 4, _MAX_TOPK, MIN_OUTSIDE
        )

        # The scalar the host *can* check is rejected rather than clamped, so a
        # caller asking for an unsupported top_k is told instead of surprised.
        with assert_raises(contains="supports top_k up to"):
            check_row(ctx, "top_k_val > _MAX_TOPK", _MAX_TOPK + 1, 32, 32)

        print("topk_softmax_sample top_k_arr contract: all cases passed")
