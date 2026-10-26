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
"""Clipped linear layers for the Gemma 4 vision tower.

A checkpoint whose vision config sets ``use_clipped_linears`` ships four
calibration bounds beside every vision projection, and its reference applies
them around the matmul::

    forward(x) = clamp(W @ clamp(x, input_min, input_max) + b,
                       output_min, output_max)

Each bound is a per-module scalar, declared as a rank-0 weight at the layer
dtype and moved to the activation's device in the graph.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.graph.quantization import QuantizationEncoding
from max.nn import MLP, Linear
from max.nn.quant_config import QuantConfig

CLAMP_BOUND_NAMES: tuple[str, ...] = (
    "input_min",
    "input_max",
    "output_min",
    "output_max",
)
"""The four calibration bounds, in checkpoint order."""


def _refuse_weight_sharding(strategy: ShardingStrategy) -> None:
    """Raises unless ``strategy`` keeps every projection whole.

    The input clamp is a function of the input alone, so any strategy applies
    it correctly. The output clamp bounds a complete matmul, and a
    weight-sharded matmul only computes partial sums, so only replicate keeps
    it meaningful.
    """
    if not strategy.is_replicate:
        raise ValueError(
            "Clipped linears shard with a replicate strategy only. The input "
            "clamp is a function of the input alone and holds under any "
            "strategy, but the output clamp bounds a complete matmul and a "
            f"weight-sharded matmul only computes partial sums. Got {strategy}."
        )


class Gemma4ClippableLinear(Linear):
    """A :class:`~max.nn.Linear` that clamps its input and its output.

    With ``use_clipping=False`` it is a plain linear that declares no extra
    weights, so an unclipped model keeps the graph and the weight names it
    had.
    """

    input_min: Weight
    input_max: Weight
    output_min: Weight
    output_max: Weight

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quant_config: QuantConfig | None = None,
        name: str | None = None,
        clip_weight: float | None = None,
        is_sharding: bool = False,
        use_clipping: bool = False,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
            quantization_encoding=quantization_encoding,
            quant_config=quant_config,
            name=name,
            clip_weight=clip_weight,
            is_sharding=is_sharding,
        )
        self.use_clipping = use_clipping

        if use_clipping and not is_sharding:
            for bound_name in CLAMP_BOUND_NAMES:
                setattr(
                    self,
                    bound_name,
                    Weight(
                        name=f"{name}.{bound_name}" if name else bound_name,
                        dtype=dtype,
                        shape=[],
                        device=DeviceRef.CPU(),
                    ),
                )

    @Linear.sharding_strategy.setter  # type: ignore[attr-defined]
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Refuses the strategies the output clamp cannot survive."""
        if self.use_clipping:
            _refuse_weight_sharding(strategy)
        Linear.sharding_strategy.fset(self, strategy)  # type: ignore[attr-defined]

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies the matmul between the input and the output clamps."""
        if not self.use_clipping:
            return super().__call__(x)

        # ops.clamp only takes Python floats, so clamp against the tensors.
        x = ops.max(x, self.input_min.cast(x.dtype).to(x.device))
        x = ops.min(x, self.input_max.cast(x.dtype).to(x.device))

        res = super().__call__(x)

        res = ops.max(res, self.output_min.cast(res.dtype).to(res.device))
        return ops.min(res, self.output_max.cast(res.dtype).to(res.device))

    # ``Linear.shard`` returns the invariant ``list[Linear]`` and
    # ``Shardable.shard`` returns the covariant ``Sequence[Self]``, so no
    # return type satisfies both bases once the element type narrows. Assert
    # the precise type here; widen ``Linear.shard`` to ``Sequence[Linear]``
    # and this ignore goes unused, which fails the type check.
    def shard(  # type: ignore[override]
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4ClippableLinear]:
        """Shards this layer, keeping the clamps.

        ``Linear.shard`` builds plain linears, which would drop them. The
        bounds are scalars, so every shard references the parent's.
        """
        devices = list(devices)
        base_shards = super().shard(devices)

        if self.use_clipping:
            # A backstop: the strategy setter refuses these strategies too.
            strategy = self.weight.sharding_strategy
            assert strategy is not None
            _refuse_weight_sharding(strategy)

        shards: list[Gemma4ClippableLinear] = []
        for base, device in zip(base_shards, devices, strict=True):
            sharded = Gemma4ClippableLinear(
                in_dim=int(self.weight.shape[1]),
                out_dim=int(self.weight.shape[0]),
                dtype=self.weight.dtype,
                device=device,
                has_bias=base.bias is not None,
                quantization_encoding=self.weight.quantization_encoding,
                quant_config=self.quant_config,
                clip_weight=self.clip_weight,
                is_sharding=True,
                use_clipping=self.use_clipping,
            )
            sharded.weight = base.weight
            sharded.bias = base.bias
            if self.use_clipping:
                for bound_name in CLAMP_BOUND_NAMES:
                    setattr(sharded, bound_name, getattr(self, bound_name))
            shards.append(sharded)

        return shards


def _builds_clipping_linears(linear_cls: Callable[..., Linear]) -> bool:
    """Whether ``linear_cls`` builds clippable linears that clamp."""
    target = linear_cls.func if isinstance(linear_cls, partial) else linear_cls
    clipped = isinstance(target, type) and issubclass(
        target, Gemma4ClippableLinear
    )
    return (
        clipped
        and getattr(linear_cls, "keywords", {}).get("use_clipping") is True
    )


class Gemma4VisionMLP(MLP):
    """The vision tower's gated MLP, with clipped projections.

    :class:`~max.nn.MLP` fuses the gate and up matmuls when it can, which
    applies neither projection's own clamp, so the fused path stays off while
    clipping is on.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear] | None = None,
        has_bias: bool = False,
        activation_function: str = "silu",
        quant_config: QuantConfig | None = None,
        swiglu_limit: float = 0.0,
        is_sharding: bool = False,
        use_clipping: bool = False,
    ) -> None:
        if use_clipping:
            if linear_cls is None:
                linear_cls = partial(Gemma4ClippableLinear, use_clipping=True)
            elif not _builds_clipping_linears(linear_cls):
                raise ValueError(
                    "Gemma4VisionMLP with use_clipping=True builds its "
                    f"projections through {linear_cls!r}, which does not "
                    "clamp. The fused gate/up matmul is off, so the MLP would "
                    "clamp nothing."
                )
        elif linear_cls is None:
            linear_cls = Linear

        super().__init__(
            dtype=dtype,
            quantization_encoding=quantization_encoding,
            hidden_dim=hidden_dim,
            feed_forward_length=feed_forward_length,
            devices=devices,
            linear_cls=linear_cls,
            has_bias=has_bias,
            activation_function=activation_function,
            quant_config=quant_config,
            swiglu_limit=swiglu_limit,
            is_sharding=is_sharding,
        )
        self.use_clipping = use_clipping

    def _can_used_fused_mlp(self) -> bool:
        """Whether the gate and up matmuls may be fused.

        A fused matmul reads the input once and so applies neither
        projection's own input clamp, so clipping keeps them apart. The base
        method is private: a rename there would silently re-enable fusion and
        stop the clamps applying, so this override is a contract on that name.
        """
        if self.use_clipping:
            return False
        return super()._can_used_fused_mlp()

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[Gemma4VisionMLP]:
        """Shards this MLP, rebuilding the shards in this class.

        ``MLP.shard`` builds plain MLPs, which would re-enable the fused path
        and drop the clamps.
        """
        devices = list(devices)
        base_shards = super().shard(devices)

        shards: list[Gemma4VisionMLP] = []
        for base, device in zip(base_shards, devices, strict=True):
            sharded = Gemma4VisionMLP(
                dtype=self.gate_proj.weight.dtype,
                quantization_encoding=self.quantization_encoding,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.feed_forward_length,
                devices=[device],
                has_bias=self.gate_proj.bias is not None,
                activation_function=self._activation_function_name,
                quant_config=self.quant_config,
                swiglu_limit=self.swiglu_limit,
                is_sharding=True,
                use_clipping=self.use_clipping,
            )
            sharded.gate_proj = base.gate_proj
            sharded.down_proj = base.down_proj
            sharded.up_proj = base.up_proj
            sharded._parent_layer = self
            shards.append(sharded)

        return shards
