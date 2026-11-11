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

The per-block gates and contraction are ``max.nn``
``HyperConnection`` (same ops as the V2 layer the graph-API model uses). The
rest follows the ModuleV2 implementation
(``deepseekV4/layers/hyper_connection.py`` at dfc8109fc0c); see its docstring.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import HyperConnection
from max.tensor import Tensor

from ..model_config import DeepseekV4Config

__all__ = ["expand_copies", "hc_head", "hc_post", "hyper_connection"]


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
