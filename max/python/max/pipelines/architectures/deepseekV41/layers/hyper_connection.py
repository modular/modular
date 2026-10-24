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

"""Hyper-connection steps around the decoder sublayers.

Between two sublayers, one fused step writes the earlier sublayer's output
back into the parallel residual streams, derives the later sublayer's mixing
gates from the new streams, and normalizes the later sublayer's input. After
the last sublayer the streams collapse into one hidden state; the model has
no dedicated weights for that step, so the last layer's FFN pre-mix weights
combine them.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue, ops
from max.nn.hyper_connection import HyperConnection
from max.nn.norm.rms_norm import RMSNorm

# Arrival counters one fused step keeps between calls.
_MEGA_MHC_COUNTERS = 1136


def fused_hc_step(
    out: TensorValue,
    residual: TensorValue,
    post: TensorValue,
    comb: TensorValue,
    pre_mix: TensorValue,
    hc: HyperConnection,
    norm: RMSNorm,
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]:
    """Joins two sublayers of the ``hc_mult`` residual streams in one step.

    Writes ``out`` back into ``residual`` through ``post`` and ``comb``,
    projects the new streams through ``hc`` for the next sublayer's gates,
    and merges the new streams with ``pre_mix`` into the next sublayer's
    normalized input. The projection runs in TF32.

    Args:
        out: ``[1, s, d]`` output of the earlier sublayer.
        residual: ``[1, s, hc, d]`` streams the earlier sublayer read.
        post: ``[1, s, hc]`` placement weights of ``out``.
        comb: ``[1, s, hc, hc]`` stream mixer of ``residual``.
        pre_mix: ``[1, s, hc]`` collapse weights carried from the earlier
            sublayer.
        hc: The later sublayer's hyper-connection weights.
        norm: The later sublayer's input norm.

    Returns:
        The later sublayer's normalized ``[1, s, d]`` input, the new
        ``[1, s, hc, d]`` streams, and the later sublayer's ``post``,
        ``comb`` and ``pre``.

    Raises:
        ValueError: If ``hc`` does not mix exactly four streams.
    """
    b, s = residual.shape[0], residual.shape[1]
    hc_mult, d = hc.hc_mult, hc.hidden_size
    if hc_mult != 4:
        raise ValueError(f"the fused step mixes 4 streams, got {hc_mult}")
    device = residual.device
    counters = ops.buffer_create(
        BufferType(DType.uint64, [_MEGA_MHC_COUNTERS], device=device),
        init_value=0,
    )
    flat_residual = ops.reshape(residual, [-1, hc_mult, d])
    rows = flat_residual.shape[0]
    streams, new_pre, new_post, new_comb, normed = ops.inplace_custom(
        "mo.mega_mhc",
        device=device,
        values=[
            ops.reshape(ops.cast(out, DType.bfloat16), [-1, d]),
            flat_residual,
            ops.reshape(post, [-1, hc_mult]),
            ops.reshape(comb, [-1, hc_mult * hc_mult]),
            ops.reshape(pre_mix, [-1, hc_mult]),
            hc.hc_fn.cast(DType.float32).to(device),
            hc.hc_scale.cast(DType.float32).to(device),
            hc.hc_base.cast(DType.float32).to(device),
            norm.weight.to(device).cast(DType.bfloat16),
            counters,
        ],
        out_types=[
            TensorType(DType.bfloat16, [rows, hc_mult, d], device),
            TensorType(DType.float32, [rows, hc_mult], device),
            TensorType(DType.float32, [rows, hc_mult], device),
            TensorType(DType.float32, [rows, hc_mult * hc_mult], device),
            TensorType(DType.bfloat16, [rows, d], device),
        ],
        parameters={
            "hidden": d,
            "shifted": True,
            "hc_norm_eps": hc.rms_norm_eps,
            "hc_pre_eps": hc.hc_eps,
            "hc_post_scale": 2.0,
            "sinkhorn_eps": hc.hc_eps,
            "sinkhorn_iters": hc.hc_sinkhorn_iters,
            "rms_norm_eps": norm.eps,
        },
    )
    return (
        ops.reshape(normed.tensor, [b, s, d]),
        ops.reshape(streams.tensor, [b, s, hc_mult, d]),
        ops.reshape(new_post.tensor, [b, s, hc_mult]),
        ops.reshape(new_comb.tensor, [b, s, hc_mult, hc_mult]),
        ops.reshape(new_pre.tensor, [b, s, hc_mult]),
    )


def final_collapse(
    full_stream: TensorValue, pre_mix: TensorValue
) -> TensorValue:
    """Weighted-sums ``hc_mult`` parallel residual streams down to one.

    Args:
        full_stream: ``[..., hc_mult, hidden_size]`` residual streams after
            the last decoder layer's FFN output is written back into them
            (not the bare FFN output).
        pre_mix: ``[..., hc_mult]`` per-stream weights: the last layer's FFN
            pre-mix.

    Returns:
        ``[..., hidden_size]``, in ``full_stream``'s dtype.
    """
    # Per-stream slices keep this one elementwise pass, with no float32 copy
    # of every stream.
    weights = ops.cast(pre_mix, DType.float32)
    collapsed = None
    for i in range(int(full_stream.shape[-2])):
        term = ops.unsqueeze(weights[..., i], -1) * ops.cast(
            full_stream[..., i, :], DType.float32
        )
        collapsed = term if collapsed is None else collapsed + term
    assert collapsed is not None
    return ops.cast(collapsed, full_stream.dtype)
