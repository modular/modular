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
"""Feed-forward sublayers for GLM-5.3-Flash: dense MLP and MoE.

Layers 0-2 are dense at ``intermediate_size`` (12288); layers 3 onward, the MTP
draft layer included, are MoE at ``moe_intermediate_size`` (2048) with one
shared expert and eight of 288 routed. Both use GLM's clamped SwiGLU.

**Every clamped-SwiGLU path already in MAX computes something else**, which is
why the dense MLP below calls :func:`clamped_swiglu` directly rather than
passing a ``limit`` to one of them:

* ``max/python/max/nn/moe/moe.py`` -- ``_swigluoai_activation`` adds GPT-OSS's
  ``+1`` bias on ``up`` and an ``alpha`` inside the sigmoid.
* ``max/python/max/nn/moe/moe.py`` -- ``make_concatenated_gated_activation_fn``
  with a ``limit`` applies the activation and *then* clamps.
* ``max/python/max/nn/linear.py`` -- the ``swiglu_limit`` branch does the same.

GLM needs ``silu(min(gate, limit)) * clamp(up, -limit, limit)``: the clamp
bounds the activation's *input*. The two orders agree below the limit and
diverge above it, so random weights never show it and a real checkpoint does.

The MoE below selects :class:`~max.nn.moe.moe.ClampedSwiGLU`, the one
custom activation the EP quantized path's fused grouped-matmul kernel
implements natively (see ``max/python/max/nn/moe/moe_fp8.py``); every other
custom ``gated_activation_fn`` is still rejected there because the kernel
needs to know the activation shape at build time.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, Value
from max.nn import MLP
from max.nn.comm.ep import EPBatchManager
from max.nn.layer import Module
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.moe.moe import ClampedSwiGLU
from max.pipelines.architectures.deepseekV3_2.layers.moe import DeepseekV3_2MoE

from .clamped_swiglu import clamped_swiglu

__all__ = [
    "Glm5NextMLP",
    "Glm5NextMlpSublayer",
    "Glm5NextMoE",
]


class Glm5NextMLP(MLP):
    """The dense MLP on layers 0-2.

    Accumulates in float32 like DeepSeek-V3.2 -- the reference evaluates
    ``silu`` on a float32 cast -- and clamps before the activation.
    """

    def __init__(self, *args, swiglu_limit: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the clamped gated MLP.

        Args:
            x: Input of shape ``(..., hidden_dim)``.

        Returns:
            Output of shape ``(..., hidden_dim)``.
        """
        x = TensorValue(x)
        gate_out = self.gate_proj(x).cast(DType.float32)
        up_out = self.up_proj(x).cast(DType.float32)
        hidden = clamped_swiglu(gate_out, up_out, self.swiglu_limit)
        return self.down_proj(hidden.cast(x.dtype))


class Glm5NextMoE(DeepseekV3_2MoE):
    """The MoE block on layers 3 onward.

    Identical to DeepSeek-V3.2's -- ``noaux_tc`` sigmoid routing with a learned
    ``e_score_correction_bias``, float32 router math, one shared expert -- apart
    from the activation, which clamps before ``silu``.

    ``n_group`` and ``topk_group`` are both 1, so group-limited routing is
    inactive and any of the 288 experts is reachable from any token. That is
    what makes expert parallelism's all-to-all dense and unbounded here, unlike
    DeepSeek-V3's node-limited routing.
    """

    def __init__(self, *args, swiglu_limit: float, **kwargs) -> None:
        kwargs["gated_activation_fn"] = ClampedSwiGLU(swiglu_limit)
        super().__init__(*args, **kwargs)
        self.swiglu_limit = swiglu_limit


class Glm5NextMlpSublayer(Module):
    """The feed-forward sublayer as the decoder layer calls it.

    Satisfies ``Glm5NextSublayer[None]``: one already-normalized sequence per
    device in, one output per device out, no residual add and no input
    layernorm -- the decoder layer owns both, because the mHC write-back is not
    an add.

    Wraps either the dense MLP or the MoE block. Its inputs bundle is the
    expert-parallel communication buffers, which an EP MoE needs and every
    other feed-forward path ignores. ``forward_moe_sharded_layers`` runs the
    full expert-parallel dispatch/compute/combine for an EP MoE and falls back
    to a plain per-shard forward for a replicated MLP, so one wrapper covers
    both.
    """

    def __init__(
        self,
        mlp: Glm5NextMLP | Glm5NextMoE,
        devices: Sequence[DeviceRef],
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.mlp = mlp
        self.ep_manager = ep_manager
        # Typed as callables rather than as the module union: that is exactly
        # what `forward_moe_sharded_layers` takes, and `shard()` returns
        # `Sequence[Self]`, which does not narrow to a union of two subclasses.
        shards: Sequence[Callable[[TensorValue], TensorValue]] = (
            mlp.shard(devices) if mlp.sharding_strategy is not None else [mlp]
        )
        self.shards = list(shards)

    @property
    def _omit_module_attr_name(self) -> bool:
        """Keeps the wrapper out of weight FQNs.

        The decoder layer holds this as ``self.mlp`` and it holds the block as
        ``self.mlp``, so without this the checkpoint's
        ``layers.N.mlp.gate_proj.weight`` would have to be
        ``layers.N.mlp.mlp.gate_proj.weight``. The wrapper exists to satisfy a
        call signature, not to add a level to the checkpoint's namespace.
        """
        return True

    def __call__(
        self, xs: list[TensorValue], inputs: Sequence[Value[Any]] = ()
    ) -> list[TensorValue]:
        """Runs the feed-forward block on every device.

        Args:
            xs: ``[total_tokens, hidden_size]`` per device, normalized.
            inputs: The expert-parallel communication buffers, empty when the
                model runs without EP. They are bound here rather than once
                for the whole stack because a subgraph body cannot reference a
                value the outer graph owns, so each MoE has to re-bind them on
                its own side of the boundary.

        Returns:
            ``[total_tokens, hidden_size]`` per device.
        """
        if self.ep_manager is not None and inputs:
            self.ep_manager.fetch_buffers(inputs)
        return forward_moe_sharded_layers(list(self.shards), xs)
