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
"""The speculative-decode accept path's per-step glue, one launch each.

Both kernels cover tensors of a few dozen entries -- `batch x draft positions`
for the rejection mask, and one scattered pair of probabilities per drafted
token for the accept test -- where a chain of whole-tensor graph ops pays a
launch floor per member. Every value is an integer or an elementwise float
with no reordering, so each folded form is bit-identical to the chain it
replaces: only the number of dispatches changes.
"""

from max.algorithm.functional import elementwise
from layout import TileTensor
from max.gpu.host import DeviceContext
from std.utils.coord import Coord


def first_rejected_index(
    rejected: TileTensor[mut=False, .bool, ...],
    dst: TileTensor[mut=True, .int64, ...],
    ctx: DeviceContext,
) raises:
    """Writes each row's first rejected draft position, `steps` if none.

    Args:
        rejected: Per-position rejection mask, `[rows, steps]`.
        dst: The first rejected position per row, `[rows, 1]`.
        ctx: The device context.

    Raises:
        Error: If the output does not carry exactly one slot per input row.
    """
    comptime assert rejected.flat_rank == 2, "rejected must be of rank 2"
    comptime assert dst.flat_rank == 2, "the destination must be of rank 2"

    var num_rows = Int(rejected.dim(0))
    var num_steps = Int(rejected.dim(1))
    var out_shape = dst.layout.shape_coord()
    if Int(out_shape[0]) != num_rows or Int(out_shape[1]) != 1:
        raise Error("first_rejected_index needs one output per input row")
    if num_rows == 0:
        return

    # A row is a serial scan of at most `num_steps` entries, so one thread owns
    # a row. `num_steps` past the last position is the "every draft accepted"
    # index -- the value a sentinel `True` column appended to the mask would
    # produce.
    @inline(.always)
    def scan[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert width == 1, "a row is scanned by one thread"
        var row = Int(idx[0])
        var first = num_steps
        for step in range(num_steps):
            if Bool(rejected.load[width=1]((row, step))):
                first = step
                break
        dst.store((row, 0), Scalar[.int64](first))

    elementwise[simd_width=1, _trace_description="sampler.first_rejected"](
        scan, (num_rows,), ctx
    )


def accept_pair(
    probs: TileTensor[mut=False, .float32, ...],
    dist: TileTensor[mut=False, .float32, ...],
    tokens: TileTensor[mut=False, .int64, ...],
    coins: TileTensor[mut=False, .float32, ...],
    rejected: TileTensor[mut=True, .bool, ...],
    fallback: TileTensor[mut=True, .int64, ...],
    ctx: DeviceContext,
) raises:
    """Writes the sampled-draft verdict's per-position pair, in one launch.

    For each ``(request, draft position)`` the accept test needs two numbers
    at one vocabulary index -- the target's masked probability there and the
    proposal's probability there -- and the residual needs to know whether the
    row carried a proposal at all. Reaching them with graph ops costs a
    `[rows, steps, 3]` index tensor to gather through, then a compare, a
    select, a multiply and a compare: one launch per member over a mask of a
    few dozen entries.

    `q == 0` means the scheduler had no distribution for the row, so the
    proposal mass is taken as one (the test degrades to typical acceptance)
    and the residual subtracts a one-hot at the drafted token instead of the
    proposal. Both the substitution and that per-row choice happen here, so
    the caller reads only the two decisions.

    Args:
        probs: The masked target distribution, `[rows, steps, vocab]`.
        dist: The proposal distribution, `[rows, steps, vocab]`.
        tokens: The drafted token per position, `[rows, steps]`.
        coins: The accept coin per position, `[rows, steps]`.
        rejected: Out, the rejection mask `[rows, steps]`.
        fallback: Out, `-1` where the row had a proposal, else its token.
        ctx: The device context.

    Raises:
        Error: If a rank or a vocabulary size disagrees with `probs`.
    """
    comptime assert probs.flat_rank == 3, "probs must be of rank 3"
    comptime assert dist.flat_rank == 3, "dist must be of rank 3"
    comptime assert tokens.flat_rank == 2, "tokens must be of rank 2"
    comptime assert coins.flat_rank == 2, "coins must be of rank 2"
    comptime assert rejected.flat_rank == 2, "rejected must be of rank 2"
    comptime assert fallback.flat_rank == 2, "fallback must be of rank 2"

    var num_rows = Int(probs.dim(0))
    var num_steps = Int(probs.dim(1))
    # Every per-position input and output is `[rows, steps]`, so a mismatch is
    # a caller bug rather than something to read around.
    if (
        Int(dist.dim(0)) != num_rows
        or Int(dist.dim(1)) != num_steps
        or Int(dist.dim(2)) != Int(probs.dim(2))
    ):
        raise Error("accept_pair needs both distributions to share a shape")
    if (
        Int(tokens.dim(0)) != num_rows
        or Int(tokens.dim(1)) != num_steps
        or Int(coins.dim(0)) != num_rows
        or Int(coins.dim(1)) != num_steps
        or Int(rejected.dim(0)) != num_rows
        or Int(rejected.dim(1)) != num_steps
        or Int(fallback.dim(0)) != num_rows
        or Int(fallback.dim(1)) != num_steps
    ):
        raise Error("accept_pair needs one entry per (request, position)")

    @inline(.always)
    def pair[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert width == 1, "one verdict per position"
        var row = Int(idx[0])
        var step = Int(idx[1])
        var token = Int(tokens.load[width=1]((row, step)))
        var p_target = probs.load[width=1]((row, step, token))
        var q_draft = dist.load[width=1]((row, step, token))
        var has_dist = Bool(q_draft > 0.0)
        # A row the scheduler had no distribution for takes q = 1, which
        # degrades the test to typical acceptance.
        var q_eff = q_draft
        if not has_dist:
            q_eff = Scalar[.float32](1.0)
        # `coin >= min(1, p / q)` without dividing, so a q that underflows to
        # zero degrades toward reject rather than accept.
        rejected.store(
            (row, step),
            Bool((coins.load[width=1]((row, step)) * q_eff) >= p_target),
        )
        # Such a row subtracts a one-hot of its drafted token from the
        # residual instead of the proposal it never carried.
        var fallback_token = Int64(-1)
        if not has_dist:
            fallback_token = Int64(token)
        fallback.store((row, step), Scalar[.int64](fallback_token))

    elementwise[simd_width=1, _trace_description="sampler.accept_pair"](
        pair, (num_rows, num_steps), ctx
    )
