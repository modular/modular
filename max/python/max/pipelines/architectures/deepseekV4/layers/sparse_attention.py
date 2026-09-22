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

"""Index-gathered sparse attention with an attention sink.

Dense-gather equivalent of ``inference/kernel.py::sparse_attn_kernel``. The
kernel fuses this into an online softmax over 64-wide index blocks; per
MXSERV-502 kernel performance is out of scope, so this gathers the selected KV
rows and runs one softmax.

Three details from the kernel that are easy to get wrong:

* **The KV is shared across heads.** ``kv`` is ``[b, n, d]``, not
  ``[b, n, h, d]`` -- V4 attention is MQA over a single latent, and the same row
  serves as both key and value.
* **``-1`` means "no such position".** The kernel writes
  ``acc_s[i, j] = -inf`` and zeroes the gathered row whenever the index is
  ``-1``. Window and compressed index construction both emit ``-1`` padding.
* **``attn_sink`` enters the denominator only.** The kernel's last steps are
  ``sum_exp[i] += exp(attn_sink[i] - scores_max[i])`` then
  ``acc_o[i, j] /= sum_exp[i]`` -- there is no matching value row, so the sink
  is a per-head escape valve that lets a query attend to "nothing". Note it is
  offset by the max over the *gathered scores only*; the sink does not
  participate in the max.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import TensorValue, ops


def sparse_attention(
    q: TensorValue,
    kv: TensorValue,
    attn_sink: TensorValue,
    topk_idxs: TensorValue,
    softmax_scale: float,
) -> TensorValue:
    """Attend from each query to its selected KV rows.

    Args:
        q: ``[batch, seq, heads, head_dim]``.
        kv: ``[batch, kv_len, head_dim]``, shared across heads.
        attn_sink: ``[heads]``, float32.
        topk_idxs: ``[batch, seq, topk]`` int32 indices into ``kv``'s axis 1;
            ``-1`` marks an unused slot.
        softmax_scale: Usually ``head_dim ** -0.5``.

    Returns:
        ``[batch, seq, heads, head_dim]`` in ``q``'s dtype.

    A query whose every index is ``-1`` produces NaN, here and in the kernel:
        the running max stays ``-inf`` and the denominator is ``exp(-inf + inf)``.
        Every query keeps at least its own position in the sliding window, so
        this does not arise.
    """
    out_dtype = q.dtype
    b, s, h, d = q.shape
    topk = topk_idxs.shape[-1]
    q32 = ops.cast(q, DType.float32)
    kv32 = ops.cast(kv, DType.float32)

    valid = topk_idxs >= 0
    safe_idxs = ops.max(
        topk_idxs, ops.constant(0, topk_idxs.dtype, topk_idxs.device)
    )

    # [b, seq, topk, head_dim]: one gather per (batch, query) row of indices.
    kv_gathered = ops.gather_nd(
        kv32, ops.unsqueeze(safe_idxs, -1), batch_dims=1
    )

    # Both matmuls run with (batch, seq) folded into one batch dim. MAX's GPU
    # batched matmul takes its tiled path once ``N % 128 == 0``, ``K % 32 == 0``
    # and ``K >= 128`` -- which the sliding-window layers reach at
    # ``seq >= 128`` (topk == window == 128) -- and on that path a fused
    # elementwise consumer (the scale, the mask, the division below) fails to
    # instantiate for a rank-4 output: ``bmm.mojo::batched_matmul_kernel_gpu``
    # hands the epilogue rank-3 coordinates, and the graph compiler's epilogue
    # rebinds them to the output's rank. Rank 3 sidesteps it. ISSUES.md Issue 32.
    rows = b * s
    q3 = ops.reshape(q32, [rows, h, d])
    kv_gathered3 = ops.reshape(kv_gathered, [rows, topk, d])

    # [b*seq, heads, topk]
    scores = ops.matmul(q3, ops.transpose(kv_gathered3, -1, -2)) * softmax_scale
    neg_inf = ops.constant(float("-inf"), DType.float32, scores.device)
    scores = ops.where(ops.reshape(valid, [rows, 1, topk]), scores, neg_inf)

    # The max is over gathered scores only -- the sink is excluded, matching
    # the kernel's reduce_max before its final sum_exp adjustment.
    scores_max = ops.max(scores, axis=-1)
    numerator = ops.exp(scores - scores_max)

    sink = ops.reshape(ops.cast(attn_sink, DType.float32), [1, h, 1])
    denominator = ops.sum(numerator, axis=-1) + ops.exp(sink - scores_max)

    out = ops.matmul(numerator, kv_gathered3) / denominator
    return ops.cast(ops.reshape(out, [b, s, h, d]), out_dtype)
