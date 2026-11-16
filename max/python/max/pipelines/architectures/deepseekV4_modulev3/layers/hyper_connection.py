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
"""mHC write-back, head contraction and expansion on ragged ``[T, hc, d]``.

On SM100 each block site is one ``hyper_connection.mega_mhc`` op
(:func:`hc_pre_norm`), as in the graph-API model: the previous sublayer's
write-back, the gates, the collapse and the sublayer's RMSNorm. A block hands
its last write-back on pending (:class:`PendingPost`) for the next block's
first site. Elsewhere a site is ``max.nn`` ``HyperConnection``. The rest
follows the ModuleV2 implementation (``deepseekV4/layers/hyper_connection.py``
at dfc8109fc0c); see its docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from max import functional as F
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType, TensorValue, ops
from max.nn import HyperConnection
from max.tensor import Tensor

from ..model_config import DeepseekV4Config
from .kernels import device_architectures
from .norm import DeepseekV4RMSNorm

__all__ = [
    "PendingPost",
    "Streams",
    "expand_copies",
    "hc_head",
    "hc_merge",
    "hc_post",
    "hc_pre_norm",
    "hyper_connection",
]


def hyper_connection(config: DeepseekV4Config) -> HyperConnection:
    """One per-block mHC site; build under a float32 default dtype."""
    return HyperConnection(
        config.hidden_size,
        config.hc_mult,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        rms_norm_eps=config.rms_norm_eps,
    )


def hc_post(x: Tensor, residual: Tensor, post: Tensor, comb: Tensor) -> Tensor:
    """``[T, d]`` sublayer output back onto the ``[T, hc, d]`` stream.

    ``out[t, k] = post[t, k] * x[t] + sum_j comb[t, j, k] * residual[t, j]``.
    Elementwise and reduced over ``j`` on purpose: ``comb^T @ residual`` would
    run the contraction in TF32 over the whole residual stream.
    """
    direct = post.unsqueeze(-1) * x.unsqueeze(-2)
    mixed = (
        (comb.unsqueeze(-1) * residual.cast(comb.dtype).unsqueeze(-2))
        .sum(axis=-3)
        .squeeze(-3)
    )
    return (direct.cast(comb.dtype) + mixed).cast(x.dtype)


def hc_head(
    x: Tensor,
    fn: Tensor,
    scale: Tensor,
    base: Tensor,
    norm_eps: float,
    hc_eps: float,
) -> Tensor:
    """``[T, hc, d]`` -> ``[T, d]`` before the final norm: sigmoid, no Sinkhorn."""
    rows, hc, width = x.shape
    flat = x.reshape([rows, hc * width]).cast(DType.float32)
    rsqrt = F.rsqrt((flat * flat).mean(axis=-1) + norm_eps)
    mixes = (flat @ fn.T) * rsqrt
    pre = F.sigmoid(mixes * scale + base) + hc_eps
    copies = flat.reshape(x.shape)
    y = (pre.unsqueeze(-1) * copies).sum(axis=-2).squeeze(-2)
    return y.cast(x.dtype)


def expand_copies(x: Tensor, hc_mult: int) -> Tensor:
    """``[T, d]`` -> ``[T, hc, d]``, every copy identical."""
    rows, width = x.shape
    return F.broadcast_to(x.unsqueeze(-2), [rows, hc_mult, width])


@dataclass(frozen=True)
class PendingPost:
    """A sublayer output not yet written back: :func:`hc_post`'s operands."""

    x: Tensor
    """``[T, d]`` sublayer output."""
    residual: Tensor
    """``[T, hc, d]`` streams the sublayer read."""
    post: Tensor
    """``[T, hc]`` float32."""
    comb: Tensor
    """``[T, hc, hc]`` float32."""


Streams: TypeAlias = Tensor | PendingPost
"""``[T, hc, d]`` streams, or a sublayer output still to be written back."""


def hc_merge(state: Streams) -> Tensor:
    """The ``[T, hc, d]`` streams, writing a pending sublayer back."""
    if isinstance(state, PendingPost):
        return hc_post(state.x, state.residual, state.post, state.comb)
    return state


# Arrival counters one ``hyper_connection.mega_mhc`` call keeps.
_MEGA_MHC_COUNTERS = 1136


def _mega_mhc(
    x: TensorValue,
    residual: TensorValue,
    post: TensorValue,
    comb: TensorValue,
    hc_fn: TensorValue,
    hc_scale: TensorValue,
    hc_base: TensorValue,
    norm_weight: TensorValue,
    hc_norm_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """``hc_post``, then the next site's gates, collapse and RMSNorm."""
    device = residual.device
    if not isinstance(device, DeviceRef):
        device = DeviceRef.from_device(device)
    rows = residual.shape[0]
    hc_mult, hidden = int(residual.shape[1]), int(residual.shape[2])
    results = ops.inplace_custom(
        "hyper_connection.mega_mhc",
        device=device,
        values=[
            ops.cast(x, DType.bfloat16),
            residual,
            post,
            ops.reshape(comb, [rows, hc_mult * hc_mult]),
            # Read only by the shifted collapse.
            post,
            ops.cast(hc_fn, DType.float32),
            ops.cast(hc_scale, DType.float32),
            ops.cast(hc_base, DType.float32),
            ops.cast(norm_weight, DType.bfloat16),
            ops.buffer_create(
                BufferType(DType.uint64, [_MEGA_MHC_COUNTERS], device=device),
                init_value=0,
            ),
        ],
        out_types=[
            TensorType(DType.bfloat16, [rows, hc_mult, hidden], device),
            TensorType(DType.float32, [rows, hc_mult], device),
            TensorType(DType.float32, [rows, hc_mult], device),
            TensorType(DType.float32, [rows, hc_mult * hc_mult], device),
            TensorType(DType.bfloat16, [rows, hidden], device),
        ],
        parameters={
            "hidden": hidden,
            "shifted": False,
            "hc_norm_eps": hc_norm_eps,
            "hc_pre_eps": hc_eps,
            "hc_post_scale": 2.0,
            "sinkhorn_eps": hc_eps,
            "sinkhorn_iters": sinkhorn_iters,
            "rms_norm_eps": norm_eps,
        },
    )
    streams, _, new_post, new_comb, normed = (r.tensor for r in results)
    return (
        streams,
        normed,
        new_post,
        ops.reshape(new_comb, [rows, hc_mult, hc_mult]),
    )


# No sharding rule: every device runs it on its whole, replicated streams.
_mega_mhc_op = F.functional(_mega_mhc)


def _mega_mhc_supported(streams: Tensor, hc: HyperConnection) -> bool:
    # The op's kernels run on SM100 and read bf16 streams, four of them, in
    # 1024-wide blocks.
    return (
        all(
            name.startswith("sm_10")
            for name in device_architectures(streams.mesh.devices)
        )
        and hc.hc_mult == 4
        and streams.dtype == DType.bfloat16
        and hc.hidden_size % 1024 == 0
    )


def hc_pre_norm(
    state: Streams, hc: HyperConnection, norm: DeepseekV4RMSNorm
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """An mHC site and its sublayer's RMSNorm: ``norm(hc(hc_merge(state)))``.

    On SM100 one ``hyper_connection.mega_mhc`` op runs the pending
    write-back and the site; its mix projection runs in TF32.

    Returns:
        The ``[T, hc, d]`` streams, the normed ``[T, d]`` sublayer input,
        ``post`` ``[T, hc]`` and ``comb`` ``[T, hc, hc]``.
    """
    if isinstance(state, PendingPost):
        x, residual, post, comb = (
            state.x,
            state.residual,
            state.post,
            state.comb,
        )
    else:
        residual = state
    if not _mega_mhc_supported(residual, hc):
        streams = hc_merge(state)
        post, comb, h = hc(streams)
        return streams, norm(h), post, comb

    if not isinstance(state, PendingPost):
        # Zero output and an identity mixer write the streams back exactly.
        rows, hc_mult, hidden = residual.shape
        mapping = residual.mapping
        x = F.zeros([rows, hidden], dtype=residual.dtype, device=mapping)
        post = F.zeros([rows, hc_mult], dtype=DType.float32, device=mapping)
        identity = [
            [float(i == j) for j in range(hc.hc_mult)]
            for i in range(hc.hc_mult)
        ]
        comb = F.broadcast_to(
            F.constant(identity, DType.float32, device=mapping),
            [rows, hc_mult, hc_mult],
        )
    outs = _mega_mhc_op(
        x,
        residual,
        post,
        comb,
        hc.hc_fn,
        hc.hc_scale,
        hc.hc_base,
        norm.weight,
        hc.rms_norm_eps,
        hc.hc_eps,
        hc.hc_sinkhorn_iters,
        norm.eps,
    )
    if residual.is_distributed:
        outs = tuple(out.rebind_mapping(residual.mapping) for out in outs)
    streams, h, post, comb = outs
    return streams, h, post, comb
