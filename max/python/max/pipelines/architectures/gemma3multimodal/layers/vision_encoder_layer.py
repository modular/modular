from collections.abc import Sequence

from max.graph import TensorValue
from max.graph.type import DeviceRef
from max.nn import LayerNorm, Module

from ..model_config import VisionConfig
from .mlp import VisionMLP
from .vision_attention import VisionAttention


class VisionEncoderLayer(Module):
    """Single transformer encoder layer for vision model."""

    def __init__(
        self,
        config: VisionConfig,
        devices: Sequence[DeviceRef],
        layer_idx: int = 0,
    ):
        """Initialize encoder layer.

        Args:
            config: Vision configuration.
            devices: Devices to place the weights on.
            layer_idx: Index of the layer (for unique naming).
        """
        super().__init__()
        self.config = config
        self.devices = devices
        self.device = devices[0] if devices else DeviceRef.CPU()
        self.layer_idx = layer_idx

        self.layer_norm1 = LayerNorm(
            dims=config.hidden_size,
            devices=devices,
            dtype=config.dtype,
            eps=config.layer_norm_eps,
            use_bias=True,
        )
        self.layer_norm1.weight.name = (
            f"encoder.layers.{layer_idx}.layer_norm1.weight"
        )
        if self.layer_norm1.bias is not None:
            self.layer_norm1.bias.name = (
                f"encoder.layers.{layer_idx}.layer_norm1.bias"
            )

        self.layer_norm2 = LayerNorm(
            dims=config.hidden_size,
            devices=devices,
            dtype=config.dtype,
            eps=config.layer_norm_eps,
            use_bias=True,
        )
        self.layer_norm2.weight.name = (
            f"encoder.layers.{layer_idx}.layer_norm2.weight"
        )
        if self.layer_norm2.bias is not None:
            self.layer_norm2.bias.name = (
                f"encoder.layers.{layer_idx}.layer_norm2.bias"
            )

        self.self_attn = VisionAttention(
            config, devices=devices, layer_idx=layer_idx
        )
        self.mlp = VisionMLP(config, devices=devices, layer_idx=layer_idx)

    def __call__(self, x: TensorValue) -> TensorValue:  # type: ignore[override]
        """Forward pass for encoder layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x
