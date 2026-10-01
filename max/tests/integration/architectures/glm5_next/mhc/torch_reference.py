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

"""Pinned torch reference for one mHC site.

Transcribed from ``DeepseekV4HyperConnection`` and
``Glm5NextTextDecoderLayer.forward`` in the transformers enablement PR
huggingface/transformers#48342 at head ``f57a815``. GLM subclasses DeepSeek-V4's
hyper-connection unchanged, so this file is the authority for both.

Two details the transcription must preserve exactly, because both are easy to
"clean up" into something that no longer matches:

* The mapping's unweighted RMSNorm uses ``rms_norm_eps``, not ``hc_eps``.
* Sinkhorn runs one column pass, then ``iters - 1`` (row, column) pairs -- 20
  column passes to 19 row passes, ending on a column pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mhc_mapping(
    streams: torch.Tensor,
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
    rms_norm_eps: float,
    mapping_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(post, comb, collapsed)`` for one site.

    ``mapping_dtype`` exists only for the fp32-vs-bf16 placebo check; the
    reference is float32 and nothing else.
    """
    hc = hc_mult
    flat = streams.flatten(start_dim=-2).to(mapping_dtype)
    flat = flat * torch.rsqrt(
        flat.to(mapping_dtype).square().mean(-1, keepdim=True) + rms_norm_eps
    )
    pre_w, post_w, comb_w = F.linear(flat, fn.to(mapping_dtype)).split(
        [hc, hc, hc * hc], dim=-1
    )
    pre_b, post_b, comb_b = base.to(mapping_dtype).split([hc, hc, hc * hc])
    pre_s, post_s, comb_s = scale.to(mapping_dtype).unbind(0)

    pre = torch.sigmoid(pre_w * pre_s + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * post_s + post_b)
    comb_logits = comb_w.view(
        *comb_w.shape[:-1], hc, hc
    ) * comb_s + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)

    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=-2).to(streams.dtype)
    return post, comb, collapsed


def mhc_write_back(
    streams: torch.Tensor,
    y: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """``streams' = post * y + comb^T @ streams``, in the model dtype."""
    dtype = streams.dtype
    return post.to(dtype).unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), streams
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """GLM's weighted RMSNorm: normalize in float32, cast, then scale."""
    normed = x.float() * torch.rsqrt(
        x.float().square().mean(-1, keepdim=True) + eps
    )
    return weight * normed.to(x.dtype)
