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

"""Multi-layer Perceptron."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.nn.legacy.float8_config import (
    Float8Config,
    Float8ScaleGranularity,
    nvfp4_packed_k,
)
from max.nn.legacy.float8_ops import matmul_float4, matmul_float8
from max.support.math import ceildiv

from .clamp import clamp
from .layer import Module, Shardable


class Linear(Module, Shardable):
    """Applies a linear transformation to incoming data: :math:`y = xW^T + b`.

    This layer implements a fully connected layer where inputs are multiplied
    by a weight matrix and optionally added with a bias vector.
    Both weights and bias initially reside on CPU, and the model init phase
    moves them to the specified device.

    Example:

    .. code-block:: python

        linear_layer = Linear(
            in_dim=256,
            out_dim=128,
            dtype=DType.float32,
            device=DeviceRef.GPU(),
            name="linear",
            has_bias=True
        )

        # Input tensor of shape: [batch, ..., 256]
        input_tensor: TensorValue
        output = linear_layer(input_tensor)
    """

    weight: Weight
    """The weight matrix stored on CPU with shape (out_dim, in_dim).
    Model init transposes the weight and moves it to the target device."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_dim,).
    Model init moves the bias to the target device if present."""

    input_scale: Weight | None = None
    """The optional input scale stored on CPU with shape ().
    Model init moves the input_scale to the target device if present."""

    weight_scale: Weight | None = None
    """The optional weight scale stored on CPU with shape () or (N,).
    Model init moves the weight_scale to the target device if present."""

    device: DeviceRef
    """The device where matrix operations are performed."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        float8_config: Float8Config | None = None,
        name: str | None = None,
        clip_weight: float | None = None,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the linear layer with weights and optional bias.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The :obj:`DType` for both weights and bias.
            device: The target :obj:`DeviceRef` for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When ``True``, adds a bias vector to the layer.
                Defaults to ``False``.
            quantization_encoding: :obj:`QuantizationEncoding` for the weights.
            float8_config: :obj:`Float8Config` for float8 quantization.
            clip_weight: Optional weight clipping threshold.
            is_sharding: Disable child layer creation during sharding.
        """
        super().__init__()

        self.device = device
        self.clip_weight = clip_weight
        self.float8_config = float8_config

        if not is_sharding:
            self.weight = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=(out_dim, nvfp4_packed_k(in_dim, float8_config)),
                device=device,
                quantization_encoding=quantization_encoding,
            )

        if has_bias:
            bias_dtype = dtype
            if float8_config and float8_config.bias_dtype:
                bias_dtype = float8_config.bias_dtype

            if not is_sharding:
                self.bias = Weight(
                    name=f"{name}.bias" if name else "bias",
                    dtype=bias_dtype,
                    shape=(out_dim,),
                    device=device,
                    quantization_encoding=quantization_encoding,
                )

                if self.bias.device != self.weight.device:
                    raise ValueError(
                        f"Bias is on device {self.bias.device} while weight is on {self.weight.device}."
                    )

        if float8_config and not is_sharding:
            if float8_config.is_static:
                self.input_scale = Weight(
                    name=f"{name}.input_scale" if name else "input_scale",
                    dtype=float8_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                    quantization_encoding=quantization_encoding,
                )

            if float8_config.input_scale.granularity not in (
                Float8ScaleGranularity.TENSOR,
                Float8ScaleGranularity.COLWISE,
                Float8ScaleGranularity.BLOCK,
            ):
                raise ValueError(
                    f"unsupported input scale granularity {float8_config.input_scale.granularity}. "
                    "Only TENSOR, COLWISE and BLOCK granularities are supported, currently"
                )

            weight_scale = float8_config.weight_scale
            weight_scale_shape = self._infer_weight_scale_shape(float8_config)

            self.weight_scale = Weight(
                name=f"{name}.weight_scale" if name else "weight_scale",
                dtype=weight_scale.dtype,
                # TODO: Pass a per-layer quantization type.
                # For now since we only support row-wise
                shape=weight_scale_shape,
                device=DeviceRef.CPU(),
                quantization_encoding=quantization_encoding,
            )
            if float8_config.is_nvfp4:
                self.weight_scale_2 = Weight(
                    name=f"{name}.weight_scale_2" if name else "weight_scale_2",
                    dtype=float8_config.input_scale.dtype,
                    shape=(),
                    device=DeviceRef.CPU(),
                    quantization_encoding=quantization_encoding,
                )

    def _infer_weight_scale_shape(
        self, float8_config: Float8Config
    ) -> tuple[int, ...]:
        weight_scale_shape: tuple[int, ...]
        weight_scale = float8_config.weight_scale
        if weight_scale.is_rowwise:
            weight_scale_shape = (int(self.weight.shape[0]), 1)
        elif weight_scale.is_tensor:
            weight_scale_shape = ()
        elif weight_scale.is_block:
            assert float8_config.weight_scale.block_size is not None
            weight_scale_shape = (
                ceildiv(
                    int(self.weight.shape[0]),
                    float8_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.weight.shape[1]),
                    float8_config.weight_scale.block_size[1],
                ),
            )
        else:
            raise ValueError(
                "only row-wise and tensor scaling are "
                f"supported currently, but got {weight_scale.granularity}"
            )
        return weight_scale_shape

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the weight sharding strategy."""
        return self.weight.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the weight sharding strategy.

        Args:
            strategy: The :obj:`ShardingStrategy` describing the weight sharding.
        """
        self.weight.sharding_strategy = strategy

        if self.weight_scale:
            # Weight scale should only be added when a float8 config is passed.
            assert self.float8_config

            # Determine weight scale sharding strategy based on weight scale type
            # and weight sharding strategy.
            if self.float8_config.weight_scale.is_tensor:
                # Tensor scaling: always replicate
                self.weight_scale.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
            elif self.float8_config.weight_scale.is_rowwise:
                if strategy.is_colwise or strategy.is_head_aware_colwise:
                    # Rowwise scale + columnwise weight: replicate to avoid shape mismatch
                    self.weight_scale.sharding_strategy = (
                        ShardingStrategy.replicate(strategy.num_devices)
                    )
                else:
                    # Rowwise scale + rowwise weight: shard along same dimension
                    self.weight_scale.sharding_strategy = strategy
            elif self.float8_config.weight_scale.is_block:
                # Block scaling: blocks correspond to regions in the weight matrix.
                # For rowwise weight sharding, shard scale's first dim (N blocks).
                # For columnwise weight sharding, shard scale's second dim (K blocks).
                if strategy.is_rowwise:
                    if (
                        int(self.weight_scale.shape[0]) % strategy.num_devices
                        != 0
                    ):
                        raise ValueError(
                            f"Weight scale dim 0 ({self.weight_scale.shape[0]}) is "
                            f"not divisible by the number of devices ({strategy.num_devices}) for block-wise scaling."
                        )
                    self.weight_scale.sharding_strategy = strategy
                elif strategy.is_colwise or strategy.is_head_aware_colwise:
                    # For columnwise weight sharding, we need to shard the scale's
                    # second dimension (K blocks) to match the sharded input.
                    if (
                        int(self.weight_scale.shape[1]) % strategy.num_devices
                        != 0
                    ):
                        raise ValueError(
                            f"Weight scale dim 1 ({self.weight_scale.shape[1]}) is "
                            f"not divisible by the number of devices ({strategy.num_devices}) for block-wise scaling with columnwise weight sharding."
                        )

                    if strategy.is_head_aware_colwise:
                        # Extract num_heads and head_dim from the partial function
                        # and compute corresponding values for the scale tensor.
                        assert isinstance(strategy.shard, partial)
                        num_heads = strategy.shard.keywords["num_heads"]
                        head_dim = strategy.shard.keywords["head_dim"]
                        # block_size is guaranteed non-None when is_block is True
                        assert (
                            self.float8_config.weight_scale.block_size
                            is not None
                        )
                        block_size_k = (
                            self.float8_config.weight_scale.block_size[1]
                        )

                        # Check if head boundaries align with block boundaries
                        if head_dim % block_size_k == 0:
                            # Each head maps to (head_dim / block_size_k) blocks
                            scale_head_dim = head_dim // block_size_k
                            self.weight_scale.sharding_strategy = (
                                ShardingStrategy.head_aware_columnwise(
                                    strategy.num_devices,
                                    num_heads,
                                    scale_head_dim,
                                )
                            )
                        else:
                            # Head boundaries don't align with blocks, fall back
                            # to even columnwise sharding.
                            self.weight_scale.sharding_strategy = (
                                ShardingStrategy.columnwise(
                                    strategy.num_devices
                                )
                            )
                    else:
                        self.weight_scale.sharding_strategy = (
                            ShardingStrategy.columnwise(strategy.num_devices)
                        )
                else:
                    self.weight_scale.sharding_strategy = strategy
            else:
                # Colwise scaling (if supported in future)
                self.weight_scale.sharding_strategy = strategy

        if self.bias:
            # Only truly shard the bias across devices when the weight sharding
            # is rowwise.
            # Otherwise, when the weight sharding is columnwise, set the bias to
            # replicate so that it is complete on device 0.
            # Linear.shard handles setting bias to None on devices >= 1 to
            # prevent bias duplication, which would be incorrect.
            self.bias.sharding_strategy = (
                strategy
                if strategy.is_rowwise
                else ShardingStrategy.replicate(strategy.num_devices)
            )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Linear]:
        """Creates sharded views of this Linear layer across multiple devices.

        Args:
            devices: Iterable of :obj:`DeviceRef` devices to place the shards on.

        Returns:
            List of sharded :obj:`Linear` instances, one for each device.
        """
        if not self.weight.sharding_strategy:
            raise ValueError(
                "Linear layer cannot be sharded because no sharding strategy was provided."
            )

        # Calculate sharded dimensions.
        out_dim = (
            int(self.weight.shape[0])
            // self.weight.sharding_strategy.num_devices
            if self.weight.sharding_strategy.is_rowwise
            else int(self.weight.shape[0])
        )

        # Get sharded weights
        sharded_weights = self.weight.shard(devices)
        sharded_biases = []
        sharded_weight_scales = []

        if self.bias is not None:
            sharded_biases = self.bias.shard(devices)

        if (
            self.float8_config
            and self.weight_scale is not None
            and len(self.weight_scale.shape) > 0
        ):
            sharded_weight_scales = self.weight_scale.shard(devices)

        shards = []
        for shard_idx, (device, weight_shard) in enumerate(
            zip(devices, sharded_weights, strict=True)
        ):
            # Create new Linear with same configuration.
            sharded = Linear(
                in_dim=int(self.weight.shape[1]),
                out_dim=out_dim,
                dtype=self.weight.dtype,
                device=device,
                has_bias=self.bias is not None,
                float8_config=self.float8_config,
                clip_weight=self.clip_weight,
                is_sharding=True,
            )

            # Replace the weights with sharded versions.
            sharded.weight = weight_shard

            # Handle bias sharding
            if self.bias is not None:
                # For columnwise sharding with allreduce.sum, only add bias on device 0
                # to avoid adding it multiple times.
                is_colwise = (
                    self.weight.sharding_strategy.is_colwise
                    or self.weight.sharding_strategy.is_head_aware_colwise
                )
                if is_colwise and (shard_idx > 0):
                    sharded.bias = None
                else:
                    sharded.bias = sharded_biases[shard_idx]

            # Handle float8 scales.
            if self.float8_config:
                if self.input_scale is not None:
                    # Input scale is always shared (scalar), which should be
                    # checked upstream.
                    assert len(self.input_scale.shape) == 0
                    sharded.input_scale = self.input_scale

                if self.weight_scale is not None:
                    # Share a reference to the original weight scale if scalar, and
                    # shard if on device.
                    # This is because scalars are always on CPU by convention.
                    sharded.weight_scale = (
                        self.weight_scale
                        if len(self.weight_scale.shape) == 0
                        else sharded_weight_scales[shard_idx]
                    )
                if (
                    self.float8_config is not None
                    and self.float8_config.is_nvfp4
                    and hasattr(self, "weight_scale_2")
                ):
                    sharded.weight_scale_2 = self.weight_scale_2

            shards.append(sharded)

        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies a linear transformation to the input data.

        Args:
            x: Input :obj:`TensorValue` of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensor must reside on the target device.

        Returns:
            Output :obj:`TensorValue` of shape ``(..., out_dim)``.
            The result resides on the target device.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        weight: TensorValue = self.weight.to(x.device)
        if self.clip_weight:
            weight = clamp(weight, -self.clip_weight, self.clip_weight)

        if self.weight.quantization_encoding:
            res = ops.qmatmul(
                self.weight.quantization_encoding, None, x, weight
            )
        elif self.float8_config:
            assert self.weight_scale is not None
            weight_scale: TensorValue = self.weight_scale

            if self.float8_config.is_nvfp4:
                assert self.input_scale is not None
                assert self.weight_scale_2 is not None
                res = matmul_float4(
                    x,
                    self.weight,
                    weight_scale,
                    self.input_scale,
                    self.weight_scale_2,
                    self.float8_config,
                )
            else:
                res = matmul_float8(
                    x,
                    self.weight,
                    weight_scale,
                    self.input_scale,
                    self.float8_config,
                )
        else:
            res = x @ weight.T

        if self.bias is not None:
            res += self.bias.to(res.device)
        return res


class ColumnParallelLinear(Linear):
    """A :obj:`Linear` layer where the weight and bias are sharded onto multiple devices.

    This layer first computes :math:`y = xW_i^T + b_i` for each device `i` in
    `[0,..., num_devices]`:

    .. code-block::

        +-----+       +-----+ T     +-----+       +-----+
        |     |       | W_0 |       | b_0 |       | y_0 | GPU0
        |     |       +-----+       +-----+       +-----+
        |     |       | W_1 |       | b_1 |       | y_1 | GPU1
        |  x  |   @   +-----+   +   +-----+   =   +-----+
        |     |       | W_2 |       | b_2 |       | y_2 | GPU2
        |     |       +-----+       +-----+       +-----+
        |     |       | W_3 |       | b_3 |       | y_3 | GPU3
        +-----+       +-----+       +-----+       +-----+

    The values are then collected using an Allgather op, producing the same
    output tensor :math:`y = xW^T + b` on each device:

    .. code-block::

        GPU0  GPU1  GPU2  GPU3                      GPU0  GPU1  GPU2  GPU3
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        | y_0 |  -  |  -  |  -  |                   | y_0 | y_0 | y_0 | y_0 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        |  -  | y_1 |  -  |  -  |                   | y_1 | y_1 | y_1 | y_1 |
        +-----+-----+-----+-----+  -- Allgather --> +-----+-----+-----+-----+
        |  -  |  -  | y_2 |  -  |                   | y_2 | y_2 | y_2 | y_2 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+
        |  -  |  -  |  -  | y_3 |                   | y_3 | y_3 | y_3 | y_3 |
        +-----+-----+-----+-----+                   +-----+-----+-----+-----+

    Example usage:

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef
        from max.nn import ColumnParallelLinear

        num_devices = 4
        distributed_linear = ColumnParallelLinear(
            in_dim,
            out_dim,
            DType.float32,
            devices=[DeviceRef.GPU(i) for i in range(num_devices)],
        )
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        devices: Sequence[DeviceRef],
        tied_weight: Weight | None = None,
        **kwargs,
    ) -> None:
        """Initializes the column-parallel linear layer.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The :obj:`DType` for both weights and bias.
            devices: The target :obj:`DeviceRef` devices for computation.
                Weights remain on CPU until sharded and moved to device during
                computation.
            tied_weight: Optional :obj:`Weight` to tie with this layer.
            **kwargs: Additional keyword arguments passed to the Linear initializer.
        """
        if len(devices) == 0:
            raise ValueError(
                "ColumnParallelLinear requires a non-empty devices argument"
            )

        if tied_weight and (
            kwargs.get("float8_config") is not None
            or kwargs.get("has_bias", False)
        ):
            raise ValueError(
                "float8 and bias are both unsupported by "
                "ColumnParallelLinear currently"
            )

        super().__init__(in_dim, out_dim, dtype, devices[0], **kwargs)

        if tied_weight:
            # Overwrite the weight we just constructed with the tied weight.
            # In contrast with overriding outside the constructor, this ensures
            # that the sharding strategy captures the tied weight correctly.
            self.weight = tied_weight
            self.set_shared_weight("weight", tied_weight)

        self.devices = devices
        self.num_devices = len(self.devices)

        self.sharding_strategy = ShardingStrategy.rowwise(self.num_devices)

        self.distributed_linear_layers = super().shard(self.devices)

    def __call__(  # type: ignore[override]
        self, x: Sequence[TensorValue], signal_buffers: Iterable[BufferValue]
    ) -> list[TensorValue]:
        """Applies a linear transformation to the input data.

        Args:
            x: Input sequence of :obj:`TensorValue` tensors of shape ``(..., in_dim)``.
                The last dimension must match the layer's ``in_dim``.
                The input tensors must reside on their respective devices.
            signal_buffers: :obj:`BufferValue` buffers for peer-to-peer communication in allgather.

        Returns:
            List of output :obj:`TensorValue` tensors of shape ``(..., out_dim)``.
            The results reside on their respective devices.

        Raises:
            ValueError: If the last dimension of ``x`` doesn't match ``in_dim``.
        """
        linear_outs = [
            self.distributed_linear_layers[i](x[i])
            for i in range(self.num_devices)
        ]
        return ops.allgather(linear_outs, signal_buffers, axis=-1)


@dataclass
class GPTQLinear(Linear):
    """A :obj:`Linear` layer for GPTQ encoding."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        quantization_encoding: QuantizationEncoding | None = None,
        quantization_config: QuantizationConfig | None = None,
        float8_config: Float8Config | None = None,
    ) -> None:
        """Initializes the linear layer with weights and optional bias with GPTQ quantization.

        Initializes the layer for GPTQ quantized linear transformations.

        Args:
            in_dim: The dimensionality of the input space.
            out_dim: The dimensionality of the output space.
            dtype: The :obj:`DType` for both weights and bias.
            device: The target :obj:`DeviceRef` for computation.
                Weights remain on CPU until moved during computation.
            has_bias: When ``True``, adds a bias vector to the layer.
                Defaults to ``False``.
            quantization_encoding: The :obj:`QuantizationEncoding` of the weights.
            quantization_config: Extra :obj:`QuantizationConfig` for the weight quantization.
            float8_config: :obj:`Float8Config` for float8 quantization (not supported).
        """
        del out_dim, dtype  # Unused.
        if has_bias:
            raise ValueError("has_bias=True is not supported in GPTQLinear.")
        if float8_config:
            raise ValueError("Float8 is not supported in GPTQLinear.")

        # Skip Linear initialization.
        Module.__init__(self)
        self.device = device
        self.qweight = Weight(
            name="qweight",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=device,
            quantization_encoding=quantization_encoding,
        )
        self.scales = Weight(
            name="scales",
            dtype=DType.uint8,
            shape=(1, 1),  # Shape will be overridden at load_state_dict.
            device=device,
            quantization_encoding=quantization_encoding,
        )

        assert quantization_config, (
            "QuantizationConfig must be provided for GPTQLinear"
        )
        assert quantization_config.sym, "GPTQ with sym=False is not supported."

        self.quantization_config = quantization_config

        desc_act = self.quantization_config.desc_act
        self.perm_idx = None
        if desc_act:
            self.perm_idx = Weight(
                "perm_idx", DType.int32, [in_dim], device=device
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies the GPTQLinear transformation to the input.

        Args:
            x: Input tensor to transform.

        Returns:
            The transformed tensor after applying the GPTQ linear layer.
        """
        assert self.qweight.quantization_encoding is not None
        qweight_dtype, qweight_shape = self.qweight.original_dtype_and_shape
        qweight = ops.reshape(
            self.qweight,
            (qweight_shape[0] * qweight_dtype.size_in_bytes, qweight_shape[1]),
        ).transpose(0, 1)

        scales_dtype, scales_shape = self.scales.original_dtype_and_shape
        scales = ops.reshape(
            self.scales,
            (scales_shape[0] * scales_dtype.size_in_bytes, scales_shape[1]),
        ).transpose(0, 1)
        weight = ops.concat((qweight, scales), axis=1).transpose(0, 1)
        if self.device:
            weight = weight.to(self.device)
        if self.perm_idx is not None:
            perm_idx: TensorValue = self.perm_idx
            if self.device:
                perm_idx = perm_idx.to(self.device)
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                ops.gather(x, perm_idx, axis=(x.rank - 1)),
                weight,
                perm_idx,
            )
        else:
            res = ops.qmatmul(
                self.qweight.quantization_encoding,
                self.quantization_config,
                x,
                weight,
            )
        if self.bias is not None:
            res += TensorValue(self.bias)
        return res


_ACTIVATION_FUNCTIONS = {
    "silu": ops.silu,
    "gelu": ops.gelu,
    "gelu_tanh": partial(ops.gelu, approximate="tanh"),
    "relu": ops.relu,
    "tanh": ops.tanh,
    "sigmoid": ops.sigmoid,
}


@dataclass
class DistributedGemmConfig:
    """Configure how distributed GEMM is executed.

    Configuration for distributed General Matrix Multiply operations.
    """

    enable_matmul_allreduce: bool
    """If ``True``, use the matmul + all_reduce kernel."""

    @staticmethod
    def generate() -> DistributedGemmConfig | None:
        """Returns the default :obj:`DistributedGemmConfig`.

        Returns:
            A :obj:`DistributedGemmConfig` instance with default settings.
        """
        opts_env = os.getenv("LLAMA_ENABLE_DIST_GEMM_KERNELS")
        if opts_env is None:
            return DistributedGemmConfig(True)

        enable_matmul_allreduce = bool(opts_env)
        return DistributedGemmConfig(enable_matmul_allreduce)


class MLP(Module, Shardable):
    """Simple multi-layer perceptron composed of three :obj:`Linear` layers.

    Defaults to SiLU activation function.
    """

    def __init__(
        self,
        dtype: DType,
        quantization_encoding: QuantizationEncoding | None,
        hidden_dim: int,
        feed_forward_length: int,
        devices: Sequence[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        has_bias: bool = False,
        activation_function: str = "silu",
        float8_config: Float8Config | None = None,
        dist_gemm_config: DistributedGemmConfig | None = None,
        is_sharding: bool = False,
    ) -> None:
        """Initializes the MLP layer.

        Args:
            dtype: :obj:`DType` to use for the layer weights, which should match the
                input dtype.
            quantization_encoding: :obj:`QuantizationEncoding` of the layer weights.
            hidden_dim: The last dimension of the layer input.
            feed_forward_length: Size of dimension used to project the inputs.
            linear_cls: :obj:`Linear` class to use to create the projection layers.
            devices: :obj:`DeviceRef` devices to run the ``MLP`` layer.
            has_bias: Whether to include bias terms in the linear layers.
            activation_function: Activation function to use. Options are:

                - ``silu``
                - ``gelu``
                - ``gelu_tanh``
                - ``relu``
                - ``tanh``
                - ``sigmoid``

            float8_config: :obj:`Float8Config` for float8 quantization.
            dist_gemm_config: :obj:`DistributedGemmConfig` for distributed GEMM configuration.
            is_sharding: Disable child layer creation during sharding.
        """
        super().__init__()
        self.devices = devices
        self.num_devices = len(devices)
        self.dist_gemm_config = dist_gemm_config
        self.hidden_dim = hidden_dim
        self.feed_forward_length = feed_forward_length

        if not is_sharding:
            self.gate_proj = linear_cls(  # [ffl, hidden]
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=devices[0],
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                float8_config=float8_config,
            )
            self.down_proj = linear_cls(
                in_dim=feed_forward_length,
                out_dim=hidden_dim,
                dtype=dtype,
                device=devices[0],
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                float8_config=float8_config,
            )
            self.up_proj = linear_cls(
                in_dim=hidden_dim,
                out_dim=feed_forward_length,
                dtype=dtype,
                device=devices[0],
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                float8_config=float8_config,
            )

        self.quantization_encoding = quantization_encoding
        self.float8_config = float8_config
        assert activation_function in _ACTIVATION_FUNCTIONS
        self._activation_function_name = activation_function
        self.activation_function = _ACTIVATION_FUNCTIONS[activation_function]
        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the MLP transformation to the input.

        Args:
            x: Input tensor to transform.

        Returns:
            The transformed tensor after applying the MLP layers.
        """
        if self.quantization_encoding or self.float8_config:
            return self.down_proj(
                self.activation_function(self.gate_proj(TensorValue(x)))
                * self.up_proj(TensorValue(x))
            )
        else:
            # Optimization to compute a single matmul by merging the
            # gate and up projection weights.
            feed_forward_length = self.gate_proj.weight.shape[0]
            gate_proj_weight: TensorValue = self.gate_proj.weight
            if self.gate_proj.device:
                gate_proj_weight = gate_proj_weight.to(self.gate_proj.device)
            up_proj_weight: TensorValue = self.up_proj.weight
            if self.up_proj.device:
                up_proj_weight = up_proj_weight.to(self.up_proj.device)

            bias = None
            if (
                self.gate_proj.bias is not None
                and self.up_proj.bias is not None
            ):
                gate_proj_bias: TensorValue = self.gate_proj.bias
                if self.gate_proj.device:
                    gate_proj_bias = gate_proj_bias.to(self.gate_proj.device)
                up_proj_bias: TensorValue = self.up_proj.bias
                if self.up_proj.device:
                    up_proj_bias = up_proj_bias.to(self.up_proj.device)
                bias = ops.concat((gate_proj_bias, up_proj_bias))

            if bias is not None:
                output = (
                    x @ ops.concat((gate_proj_weight, up_proj_weight)).T
                ) + bias
            else:
                output = x @ ops.concat((gate_proj_weight, up_proj_weight)).T

            gate_out, up_out = ops.split(
                output, [feed_forward_length, feed_forward_length], axis=1
            )

            hidden = self.activation_function(gate_out) * up_out
            # If we overlap GEMM / AllReduce, the last linear layer is skipped.
            if (
                self.dist_gemm_config is None
                or not self.dist_gemm_config.enable_matmul_allreduce
            ):
                return self.down_proj(hidden)
            else:
                return hidden

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the MLP sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the MLP layers.

        Args:
            strategy: The sharding strategy to apply.
        """
        self._sharding_strategy = strategy

        if strategy.is_replicate:
            # For replicate strategy, both layers use the same strategy
            self.gate_proj.sharding_strategy = strategy
            self.down_proj.sharding_strategy = strategy
            self.up_proj.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.gate_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.down_proj.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
            self.up_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
        else:
            raise ValueError(f"Unsupported sharding strategy: {strategy}")

    def shard(self, devices: Iterable[DeviceRef]) -> list[MLP]:
        """Creates sharded views of this MLP across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MLP instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        # Get sharded layers
        sharded_gate_projs = self.gate_proj.shard(devices)
        sharded_down_projs = self.down_proj.shard(devices)
        sharded_up_projs = self.up_proj.shard(devices)

        shards = []
        for device, gate_proj, down_proj, up_proj in zip(
            devices,
            sharded_gate_projs,
            sharded_down_projs,
            sharded_up_projs,
            strict=True,
        ):
            # Create new MLP instance with the sharded layers
            sharded = MLP(
                dtype=self.gate_proj.weight.dtype,
                quantization_encoding=self.quantization_encoding,
                hidden_dim=self.hidden_dim,
                feed_forward_length=self.feed_forward_length,
                devices=[device],
                has_bias=self.gate_proj.bias is not None,
                activation_function=self._activation_function_name,
                float8_config=self.float8_config,
                dist_gemm_config=self.dist_gemm_config,
                is_sharding=True,
            )

            # Assign the sharded linear layers
            sharded.gate_proj = gate_proj
            sharded.down_proj = down_proj
            sharded.up_proj = up_proj

            shards.append(sharded)

        return shards
