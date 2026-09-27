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
"""Split and Sinkhorn projection of manifold-constrained hyper-connections.

Each token's ``(2 + hc) * hc`` mixing logits split into a ``pre`` read weight,
a ``post`` write weight and an ``hc x hc`` combination matrix. The combination
is row-softmaxed and then alternately column- and row-normalized
``sinkhorn_iters`` times, which drives it toward doubly stochastic.

Written as graph ops the normalizations are a serial chain of two tiny kernels
each, ~80 launches per site for a 4x4 matrix. Here one thread owns one token
and runs the whole chain in registers, so a site is a single launch.
"""

from std.math import exp, max

from layout import Coord, TileTensor
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.sys import align_of

from nn.activations import sigmoid


@inline(.always)
def _col_norm[hc: Int](mut m: SIMD[DType.float32, hc * hc], eps: Float32):
    comptime for j in range(hc):
        var s = Float32(0)
        comptime for i in range(hc):
            s += m[i * hc + j]
        var d = s + eps
        comptime for i in range(hc):
            m[i * hc + j] = m[i * hc + j] / d


@inline(.always)
def _row_norm[hc: Int](mut m: SIMD[DType.float32, hc * hc], eps: Float32):
    comptime for i in range(hc):
        var s = Float32(0)
        comptime for j in range(hc):
            s += m[i * hc + j]
        var d = s + eps
        comptime for j in range(hc):
            m[i * hc + j] = m[i * hc + j] / d


def mhc_split_sinkhorn[
    hc: Int,
    target: StaticString,
](
    pre: TileTensor[mut=True, DType.float32, ...],
    post: TileTensor[mut=True, DType.float32, ...],
    comb: TileTensor[mut=True, DType.float32, ...],
    mixes: TileTensor[DType.float32, ...],
    scale: TileTensor[DType.float32, ...],
    base: TileTensor[DType.float32, ...],
    sinkhorn_iters: Int,
    eps: Float32,
    post_mult: Float32,
    ctx: DeviceContext,
) raises:
    """Splits ``mixes`` into ``pre``, ``post`` and the Sinkhorn'd ``comb``.

    Matches the reference ``hc_split_sinkhorn``: with ``m = mixes[t]``,

    * ``pre[t, k] = sigmoid(m[k] * scale[0] + base[k]) + eps``
    * ``post[t, k] = post_mult * sigmoid(m[hc + k] * scale[1] + base[hc + k])``
    * ``comb[t]`` is ``m[2 hc:] * scale[2] + base[2 hc:]`` as a row-major
      ``hc x hc`` matrix, row-softmaxed plus ``eps``, column-normalized, then
      ``sinkhorn_iters - 1`` rounds of row then column normalization, each
      dividing by ``sum + eps``.

    Parameters:
        hc: The number of residual copies.
        target: Compilation target string, selects the CPU or GPU path.

    Args:
        pre: ``[tokens, hc]`` output.
        post: ``[tokens, hc]`` output.
        comb: ``[tokens, hc * hc]`` output, row-major ``hc x hc`` per token.
        mixes: ``[tokens, (2 + hc) * hc]`` mixing logits.
        scale: ``[3]`` per-part scales for pre, post and comb.
        base: ``[(2 + hc) * hc]`` per-output bias.
        sinkhorn_iters: Row/column normalization rounds, at least one.
        eps: Added to ``pre``, to the softmax and to every normalizer.
        post_mult: Multiplier on the ``post`` sigmoid.
        ctx: Device context used to enqueue the kernel.
    """
    comptime assert pre.flat_rank == 2 and post.flat_rank == 2
    comptime assert comb.flat_rank == 2 and mixes.flat_rank == 2
    comptime assert scale.flat_rank == 1 and base.flat_rank == 1
    comptime nn_ = hc * hc
    comptime assert nn_ & (nn_ - 1) == 0, "hc * hc must be a power of two"
    comptime off_comb = 2 * hc

    var tokens = Int(mixes.dim[0]())
    if tokens == 0:
        return
    if sinkhorn_iters < 1:
        raise Error("mhc_split_sinkhorn: sinkhorn_iters must be >= 1")

    @inline(.always)
    def token_fn[width: Int, alignment: Int = 1](idx: Coord) {var}:
        comptime assert idx.rank == 1
        var t = Int(idx[0].value())

        var s_pre = scale.load[width=1]((0,))
        var s_post = scale.load[width=1]((1,))
        var s_comb = scale.load[width=1]((2,))

        comptime for k in range(hc):
            var p = mixes.load[width=1]((t, k))
            pre.store[width=1](
                (t, k), sigmoid(p * s_pre + base.load[width=1]((k,))) + eps
            )
            var q = mixes.load[width=1]((t, hc + k))
            post.store[width=1](
                (t, k),
                post_mult * sigmoid(q * s_post + base.load[width=1]((hc + k,))),
            )

        # Rows are (2 + hc) * hc floats, so the comb block is only
        # element-aligned.
        comptime align = align_of[Float32]()
        var c = mixes.load[width=nn_, alignment=align](
            (t, off_comb)
        ) * s_comb + base.load[width=nn_, alignment=align]((off_comb,))

        comptime for i in range(hc):
            var row_max = c[i * hc]
            comptime for j in range(1, hc):
                row_max = max(row_max, c[i * hc + j])
            var row_sum = Float32(0)
            comptime for j in range(hc):
                var v = exp(c[i * hc + j] - row_max)
                c[i * hc + j] = v
                row_sum += v
            var recip = Float32(1) / row_sum
            comptime for j in range(hc):
                c[i * hc + j] = c[i * hc + j] * recip + eps

        _col_norm[hc](c, eps)
        for _ in range(sinkhorn_iters - 1):
            _row_norm[hc](c, eps)
            _col_norm[hc](c, eps)

        comb.store[width=nn_, alignment=align]((t, 0), c)

    elementwise[
        simd_width=1,
        target=target,
        _trace_description="mhc_split_sinkhorn",
    ](token_fn, (tokens,), ctx)
