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
"""The GLM-5.3-Flash decoder layer, on ModuleV3.

Manifold-constrained hyper-connections replace the residual add, so this layer
owns the residual arithmetic: each of its two sites collapses the
``[tokens, hc_mult, hidden]`` streams into the sublayer's input and writes the
sublayer's output back into all of them.
"""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import HyperConnection, Module, RMSNorm
from max.experimental.nn.common_layers.quant_moe import ExpertParallelMoE
from max.experimental.sharding import DeviceMapping, Sharded, auto_reshard
from max.experimental.tensor import Tensor
from max.nn.comm.ep import EPCommBuffers

from ...deepseekV3_2_modulev3.layers.moe import DeepseekV3_2TensorParallelMoE
from ..model_config import Glm5NextConfig

__all__ = ["Glm5NextDecoderLayer", "write_back"]


def write_back(
    streams: Tensor, y: Tensor, post: Tensor, comb: Tensor
) -> Tensor:
    """Places a sublayer output back into the streams.

    ``streams' = post * y + comb^T @ streams``. ``comb``'s columns sum to one,
    so contracting over its first stream axis makes the mix a convex
    combination of the incoming streams.
    """
    dtype = streams.dtype
    placed = post.cast(dtype).unsqueeze(-1) * y.unsqueeze(1)
    mixed = comb.cast(dtype).transpose(-1, -2) @ streams
    return placed + mixed


def _site(config: Glm5NextConfig) -> HyperConnection:
    return HyperConnection(
        config.hidden_size,
        config.hc_mult,
        hc_sinkhorn_iters=config.hc_sinkhorn_iters,
        hc_eps=config.hc_eps,
        rms_norm_eps=config.rms_norm_eps,
    )


class Glm5NextDecoderLayer(Module[..., Tensor]):
    """One decoder layer: attention site, then feed-forward site.

    Args:
        config: The model config.
        self_attn: The attention sublayer, KDA or sparse MLA. Returns its
            output projection's partial sums.
        mlp: The feed-forward block, dense or MoE.
    """

    def __init__(
        self,
        config: Glm5NextConfig,
        self_attn: Module[..., Tensor],
        mlp: Module[..., Tensor],
    ) -> None:
        self.self_attn = self_attn
        self.mlp = mlp
        self.input_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_attn = _site(config)
        self.hc_ffn = _site(config)

    def forward(
        self,
        streams: Tensor,
        attn_args: tuple[object, ...],
        comm_buffers: EPCommBuffers | None,
    ) -> Tensor:
        replicated = DeviceMapping.replicated(streams.mesh)
        post, comb, collapsed = self.hc_attn(streams)
        # The attention output is a sum of per-device partials.
        y = self.self_attn(self.input_layernorm(collapsed), *attn_args)
        streams = write_back(streams, y.to(replicated), post, comb)

        post, comb, collapsed = self.hc_ffn(streams)
        h = self.post_attention_layernorm(collapsed)
        if isinstance(self.mlp, DeepseekV3_2TensorParallelMoE):
            # Its experts are split by columns, so every device needs every
            # row. Its float32 combine is a sum of per-device partials, and
            # casting it back to bfloat16 needs the sum: let that cast
            # all-reduce first.
            with auto_reshard(mode="silent"):
                y = self.mlp(h)
            return write_back(streams, y.to(replicated), post, comb)
        # The other feed-forward blocks are per token, so each device runs
        # them over its own share of the rows rather than every device over
        # all of them. That is also what keeps a device's EP dispatch under
        # its cap.
        h = h.to(DeviceMapping(h.mesh, (Sharded(0),)))
        if isinstance(self.mlp, ExpertParallelMoE):
            y = self.mlp(h, comm_buffers)
        else:
            y = self.mlp(h)
        # Gathering the shards back gives a row count the compiler cannot
        # prove equal to the streams'; it is, by construction.
        y = F.rebind(y.to(replicated), [streams.shape[0], y.shape[1]])
        return write_back(streams, y, post, comb)
