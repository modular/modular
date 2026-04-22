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

"""Stacked linear projection layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.clamp import clamp
from max.nn.kernels import convert_weights_to_fp8_fnuz_if_needed
from max.nn.layer import Module
from max.nn.linear import Linear, linear
from max.nn.quant_config import QuantConfig
from max.nn.quant_ops import quantize_static_scaled_float8


class StackedLinear(Module):
    """A module that manages multiple linear projections as a stacked weight.

    Supports two modes:

    - **Stacked** (``stacked=True``): Holds a single pre-stacked weight tensor.
      Use when the checkpoint already stores a fused weight (e.g. ``qkv_proj.weight``).
    - **Unfused** (``stacked=False``): Holds N child :class:`~max.nn.Linear`
      modules whose weights are concatenated at graph-build time.
      Use when the checkpoint stores separate projections (e.g. ``q_proj``,
      ``k_proj``, ``v_proj``).

    In **unfused** mode (``stacked=False``), the module sets
    :attr:`~max.nn.Module._omit_module_attr_name`: its own attribute name
    (typically ``qkv_proj``) is omitted from the FQN of its child weights.
    The child names supplied via the ``names`` argument therefore double
    as the external (checkpoint) names. For QKV stacking that means using
    ``names=["q_proj", "k_proj", "v_proj"]`` so that
    ``self.qkv_proj = StackedLinear(...)`` exposes weights at
    ``self_attn.q_proj.weight`` rather than
    ``self_attn.qkv_proj.q_proj.weight``. This removes the need for
    per-architecture ``q_proj -> qkv_proj.q`` mapping in weight adapters.

    In **stacked** mode (``stacked=True``), the attribute name is *not*
    omitted: the single fused ``weight``/``bias`` would otherwise lose
    all namespace context and collide with sibling attributes.
    Stacked-mode weights remain at ``<attr>.weight`` / ``<attr>.bias``
    (e.g. ``self_attn.qkv_proj.weight``) and weight adapters must
    continue to map fused checkpoint names into that namespace.
    """

    def __init__(
        self,
        in_dim: int,
        out_dims: Sequence[int],
        names: Sequence[str],
        dtype: DType,
        device: DeviceRef,
        stacked: bool = False,
        has_bias: bool = False,
        linear_cls: Callable[..., Linear] = Linear,
        quant_config: QuantConfig | None = None,
        clip_weight: float | None = None,
        _is_sharding: bool = False,
    ) -> None:
        """Initializes the stacked linear layer.

        Args:
            in_dim: The input dimension shared by all projections.
            out_dims: Output dimension for each projection.
            names: Attribute name for each child (e.g.
                ``["q_proj", "k_proj", "v_proj"]``). In unfused mode these
                names are also the FQNs the children's weights are exposed
                under (see class docstring on
                ``_omit_module_attr_name``), so they should match the
                corresponding checkpoint names.
            dtype: Data type for all weights.
            device: Device for weight placement.
            stacked: When ``True``, create a single pre-stacked weight
                instead of N child :class:`~max.nn.Linear` modules.
            has_bias: Whether each projection has a bias vector.
            linear_cls: Linear class to use for each projection.
            quant_config: Optional quantization config.
            clip_weight: Optional weight clipping threshold.
        """
        super().__init__()

        self._names = list(names)
        self._stacked = stacked
        self._out_dims = list(out_dims)
        self._in_dim = in_dim
        self._has_bias = has_bias
        self._clip_weight = clip_weight
        self._quant_config = quant_config

        if _is_sharding:
            return

        if stacked:
            if clip_weight:
                raise ValueError(
                    "clip_weight is not yet supported with stacked=True."
                )
            if quant_config and quant_config.is_static:
                raise NotImplementedError(
                    "Float8 static scaling with stacked=True is not supported"
                    " yet."
                )

            total_out = sum(out_dims)
            self.weight = Weight(
                name="weight",
                dtype=dtype,
                shape=[total_out, in_dim],
                device=device,
            )
            if has_bias:
                self.bias = Weight(
                    name="bias",
                    dtype=dtype,
                    shape=[total_out],
                    device=device,
                )
        else:
            for name, out_dim in zip(names, out_dims, strict=True):
                setattr(
                    self,
                    name,
                    linear_cls(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        dtype=dtype,
                        device=device,
                        has_bias=has_bias,
                        quant_config=quant_config,
                        clip_weight=clip_weight,
                    ),
                )

    @property
    def _omit_module_attr_name(self) -> bool:
        """Only unfused mode omits its attribute name from descendant FQNs.

        Computed from ``self._stacked`` rather than stored as instance state
        so it stays correct regardless of how the instance was constructed
        (``__init__`` vs ``__new__`` / ``copy.copy`` / custom
        deserialization). Stacked mode keeps its attribute name as the
        prefix because the single fused weight has no per-projection name
        to fall back on.
        """
        return not self._stacked

    def _child(self, name: str) -> Linear:
        return getattr(self, name)

    @property
    def stacked_weight(self) -> TensorValue:
        """Returns the stacked weight tensor.

        For stacked mode, returns the single weight directly.
        For unfused mode, delegates to :meth:`_concat_child_weights`.
        """
        if self._stacked:
            return self.weight
        return self._concat_child_weights()

    def _concat_child_weights(self) -> TensorValue:
        """Collect, transform, and concatenate child weights.

        The default implementation handles weight clipping, static FP8
        dequant-then-requant, and NVFP4 passthrough.
        """
        weights: list[TensorValue] = [
            self._child(n).weight for n in self._names
        ]

        if self._clip_weight:
            weights = [
                clamp(w, min=-self._clip_weight, max=self._clip_weight)
                for w in weights
            ]

        # For static per-tensor FP8: dequant each projection with its
        # own scale, then requant under the unified max scale.
        # Dynamic tensor-wise FP8 skips this: per-projection scales
        # are broadcast to rowwise in stacked_weight_scale instead.
        if (
            self._quant_config
            and self._quant_config.is_static
            and self._quant_config.weight_scale.is_tensor
            and all(
                (ws := self._child(n).weight_scale) is not None and ws.rank != 2
                for n in self._names
            )
        ):
            dequanted: list[TensorValue] = []
            for w, n in zip(weights, self._names, strict=True):
                ws = self._child(n).weight_scale
                assert ws is not None
                dequanted.append(w * ws.to(w.device))
            weights = dequanted

        wstacked = ops.concat(weights)

        if self._quant_config and self._quant_config.is_nvfp4:
            return wstacked
        if self._quant_config and self._quant_config.is_static:
            assert self.stacked_weight_scale is not None
            wstacked, scale = convert_weights_to_fp8_fnuz_if_needed(
                wstacked, self.stacked_weight_scale.to(DeviceRef.CPU())
            )
            wstacked = quantize_static_scaled_float8(
                wstacked, scale.to(DeviceRef.CPU())
            )

        return wstacked

    @property
    def stacked_bias(self) -> TensorValue | None:
        """Returns the concatenated bias vector, or ``None``."""
        if not self._has_bias:
            return None
        if self._stacked:
            return self.bias
        biases = []
        for n in self._names:
            b = self._child(n).bias
            assert b is not None
            biases.append(b)
        return ops.concat(biases)

    @property
    def stacked_input_scale(self) -> TensorValue | None:
        """Returns the max of per-projection input scales, or ``None``."""
        if not self._quant_config or self._quant_config.is_dynamic:
            return None
        if self._stacked:
            raise NotImplementedError(
                "Input scale not implemented for stacked=True"
            )

        scales = []
        for n in self._names:
            s = self._child(n).input_scale
            assert s is not None
            scales.append(s.reshape((1,)))
        return ops.max(ops.concat(scales)).reshape(())

    @property
    def stacked_weight_scale(self) -> TensorValue | None:
        """Returns the combined weight scale for quantized matmul."""
        if not self._quant_config:
            return None
        if self._stacked:
            raise NotImplementedError(
                "Weight scale not implemented for stacked=True"
            )

        scales: list[TensorValue] = []
        for n in self._names:
            s = self._child(n).weight_scale
            assert s is not None
            scale_val: TensorValue = s
            if len(s.shape) == 0:
                scale_val = s.reshape((1,))
            scales.append(scale_val)

        weight_scale = ops.concat(scales)

        if weight_scale.rank == 2:
            return weight_scale

        # For dynamic tensor-wise FP8: broadcast each projection's
        # scalar scale to [dim, 1] and concatenate so each row keeps
        # its exact original scale.
        if (
            self._quant_config
            and self._quant_config.weight_scale.is_tensor
            and self._quant_config.is_dynamic
        ):
            rows = []
            for n, out_dim in zip(self._names, self._out_dims, strict=True):
                s = self._child(n).weight_scale
                assert s is not None
                rows.append(ops.broadcast_to(s.reshape([1, 1]), [out_dim, 1]))
            return ops.concat(rows)

        return ops.max(weight_scale).reshape([])

    @property
    def stacked_weight_scale_2(self) -> TensorValue | None:
        """Returns the max of per-projection weight_scale_2 (NVFP4)."""
        if (
            not self._quant_config
            or self._quant_config.is_dynamic
            or not self._quant_config.is_nvfp4
        ):
            return None
        if self._stacked:
            raise NotImplementedError(
                "Weight scale 2 not implemented for stacked=True"
            )

        scales = []
        for n in self._names:
            s = self._child(n).weight_scale_2
            assert s is not None
            scales.append(s.reshape((1,)))
        return ops.max(ops.concat(scales)).reshape(())

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy."""
        if self._stacked:
            return self.weight.sharding_strategy
        return self._child(self._names[0]).sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set sharding strategy, propagating to children or the single weight."""
        if self._stacked:
            self.weight.sharding_strategy = strategy
            if self._has_bias:
                self.bias.sharding_strategy = strategy
        else:
            for n in self._names:
                self._child(n).sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[StackedLinear]:
        """Create sharded copies across devices.

        For stacked mode, shards the single weight.
        For unfused mode, shards each child Linear and reassembles.
        """
        devices = list(devices)
        num_devices = len(devices)

        # Per-shard out_dims so downstream consumers (notably
        # ``stacked_weight_scale``'s broadcast for dynamic tensor-wise
        # FP8) see shapes aligned with the sharded child weights.
        shard_out_dims = [d // num_devices for d in self._out_dims]

        if self._stacked:
            weight_shards = self.weight.shard(devices)
            bias_shards = self.bias.shard(devices) if self._has_bias else None
            shards = []
            for i, device in enumerate(devices):
                sl = StackedLinear(
                    in_dim=self._in_dim,
                    out_dims=shard_out_dims,
                    names=self._names,
                    dtype=self.weight.dtype,
                    device=device,
                    stacked=True,
                    has_bias=self._has_bias,
                    quant_config=self._quant_config,
                    clip_weight=self._clip_weight,
                    _is_sharding=True,
                )
                sl.weight = weight_shards[i]
                if bias_shards is not None:
                    sl.bias = bias_shards[i]
                shards.append(sl)
            return shards
        else:
            child_shards: dict[str, list[Linear]] = {}
            for n in self._names:
                child_shards[n] = self._child(n).shard(devices)

            # Re-use the parent's linear_cls so a re-shard wouldn't
            # silently fall back to the default ``Linear``.
            linear_cls = type(self._child(self._names[0]))

            shards = []
            for i, device in enumerate(devices):
                sl = StackedLinear(
                    in_dim=self._in_dim,
                    out_dims=shard_out_dims,
                    names=self._names,
                    dtype=self._child(self._names[0]).weight.dtype,
                    device=device,
                    stacked=False,
                    has_bias=self._has_bias,
                    linear_cls=linear_cls,
                    quant_config=self._quant_config,
                    clip_weight=self._clip_weight,
                    _is_sharding=True,
                )
                for n in self._names:
                    setattr(sl, n, child_shards[n][i])
                shards.append(sl)
            return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Computes ``x @ stacked_weight.T``, with quantization and bias."""
        w = self.stacked_weight.to(x.device)

        result = linear(
            x,
            w,
            quant_config=self._quant_config,
            input_scale=self.stacked_input_scale,
            weight_scale=self.stacked_weight_scale,
            weight_scale_2=self.stacked_weight_scale_2,
        )

        bias = self.stacked_bias
        if bias is not None:
            result = result + bias.to(x.device)

        return result
