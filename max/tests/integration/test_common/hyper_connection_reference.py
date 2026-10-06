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

"""PyTorch reference for the manifold-constrained hyper-connection layer.

Shared by the ModuleV2 (:class:`max.nn.HyperConnection`) and ModuleV3
(:class:`max.experimental.nn.HyperConnection`) tests so both are measured
against one definition of the layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hyper_connection_reference(
    hidden_streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_base: torch.Tensor,
    hc_scale: torch.Tensor,
    *,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    hc_eps: float,
    rms_norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the mHC gates and the collapsed stream in float32.

    Args:
        hidden_streams: The parallel residual streams, shaped
            ``[total_seq_len, hc_mult, hidden_size]``.
        hc_fn: The stream projection, shaped
            ``[2 * hc_mult + hc_mult ** 2, hc_mult * hidden_size]``.
        hc_base: The per-output bias, in ``pre``, ``post``, ``comb`` order.
        hc_scale: The per-output scale, in ``pre``, ``post``, ``comb`` order.
        hc_mult: The number of parallel residual streams.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project ``comb``.
        hc_eps: Epsilon guarding the Sinkhorn divisions.
        rms_norm_eps: Epsilon of the unweighted RMSNorm applied to the streams.

    Returns:
        A tuple ``(post, comb, collapsed)``. ``comb`` is
        ``[total_seq_len, hc_mult, hc_mult]``; ``collapsed`` carries
        ``hidden_streams``' dtype.
    """
    hc = hc_mult
    streams = hidden_streams.float()

    flat = streams.flatten(start_dim=1)
    flat = flat * torch.rsqrt(flat.pow(2).mean(-1, keepdim=True) + rms_norm_eps)

    proj = F.linear(flat, hc_fn.float())
    pre_w, post_w, comb_w = proj.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = hc_base.float().split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = hc_scale.float().unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)

    comb_logits = comb_w.view(
        *comb_w.shape[:-1], hc, hc
    ) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1)
    return post, comb, collapsed.to(hidden_streams.dtype)
