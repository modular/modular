from __future__ import annotations

from collections.abc import Sequence

from max.dtype.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    TensorValue,
    Weight,
)
from max.nn import LayerList, LayerNorm, Module
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm

from .language_model import Gemma3LanguageModelWithVision
from .layers.vision_embeddings import VisionEmbeddings
from .layers.vision_encoder_layer import VisionEncoderLayer
from .model_config import Gemma3MultimodalConfig, VisionConfig


class VisionEncoder(Module):
    """Stack of vision transformer encoder layers for SigLip."""

    def __init__(
        self,
        config: VisionConfig,
        devices: Sequence[DeviceRef],
    ):
        """Initialize vision encoder.

        Args:
            config: Vision configuration.
            devices: Devices to place the weights on.
        """
        super().__init__()
        self.config = config
        self.devices = devices

        self.layers = LayerList(
            [
                VisionEncoderLayer(config, devices=devices, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )

    def __call__(self, x: TensorValue, output_hidden_states: bool = False):  # type: ignore[override]
        """Forward pass through all encoder layers.

        Args:
            x: Input tensor of shape (batch_size, seq_len, hidden_size).
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            If output_hidden_states is False:
                Output tensor of shape (batch_size, seq_len, hidden_size).
            If output_hidden_states is True:
                Tuple of (output_tensor, hidden_states_list).
        """
        hidden_states = []

        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        if output_hidden_states:
            return x, hidden_states
        else:
            return x


class SigLipVisionModel(Module):
    """SigLip Vision Model for Gemma3 multimodal.

    This implements the vision encoder component that processes images
    into embeddings that can be integrated with the language model.

    The architecture follows the SigLip model structure:
    - Patch embedding via Conv2d (patch_size x patch_size)
    - Position embeddings
    - Multi-layer transformer encoder
    - Final layer normalization

    Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma3/vision.py
    """

    def __init__(
        self, config: VisionConfig, devices: Sequence[DeviceRef] | None = None
    ):
        """Initialize the SigLip vision model.

        Args:
            config: Vision configuration containing model hyperparameters.
            devices: Devices to place the weights on.
        """
        super().__init__()
        self.config = config
        self.devices = devices or [DeviceRef.CPU()]
        self.device = self.devices[0]
        self.embeddings = VisionEmbeddings(config, devices=self.devices)
        self.encoder = VisionEncoder(config, devices=self.devices)

        self.post_layernorm = LayerNorm(
            dims=config.hidden_size,
            devices=self.devices,
            dtype=config.dtype,
            eps=config.layer_norm_eps,
            use_bias=True,
        )
        self.post_layernorm.weight.name = "post_layernorm.weight"
        if self.post_layernorm.bias is not None:
            self.post_layernorm.bias.name = "post_layernorm.bias"

    def __call__(
        self, pixel_values: TensorValue, output_hidden_states: bool = False
    ):  # type: ignore[override]
        """Forward pass for vision model.

        Args:
            pixel_values: Input images of shape (batch_size, num_channels, height, width).
            output_hidden_states: Whether to return hidden states from all layers.

        Returns:
            If output_hidden_states is False:
                pooler_output: Final output of shape (batch_size, num_patches, hidden_size).
            If output_hidden_states is True:
                Tuple of (pooler_output, embeddings, hidden_states).
        """
        # Get patch embeddings
        hidden_states = self.embeddings(pixel_values)
        embeddings = hidden_states

        # Pass through encoder
        encoder_output = self.encoder(
            hidden_states, output_hidden_states=output_hidden_states
        )

        if output_hidden_states:
            final_output, hidden_states_list = encoder_output
        else:
            final_output = encoder_output
            hidden_states_list = []

        # Apply final layer norm
        pooler_output = self.post_layernorm(final_output)

        if output_hidden_states:
            return (pooler_output, embeddings, hidden_states_list)
        else:
            return pooler_output


class MultimodalProjector(Module):
    """Projects vision embeddings to language model hidden size with spatial pooling."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        dtype: DType,
        device: DeviceRef,
        image_size: int = 896,
        patch_size: int = 14,
        mm_tokens_per_image: int = 256,
    ):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.device = device
        self.dtype = dtype

        self.patches_per_side = image_size // patch_size
        self.tokens_per_side = int(mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_side // self.tokens_per_side

        # Soft embedding normalization applied to vision embeddings before projection
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            dim=vision_hidden_size,
            dtype=dtype,
            eps=1e-6,
        )

        self.mm_input_projection_weight = Weight(
            name="mm_input_projection_weight",
            dtype=dtype,
            shape=[vision_hidden_size, text_hidden_size],
            device=device,
        )

    def __call__(self, vision_embeddings: TensorValue) -> TensorValue:
        """Project vision embeddings to text hidden size with spatial pooling.

        Args:
            vision_embeddings: Vision embeddings of shape (batch*num_patches, vision_hidden_size)

        Returns:
            Projected embeddings of shape (batch*256, text_hidden_size) after pooling
        """
        from max.graph import ops
        from max.graph.ops import avg_pool2d

        total_embeddings = vision_embeddings.shape[0]
        total_patches = self.patches_per_side * self.patches_per_side
        batch_size = total_embeddings // total_patches

        vision_embeddings_batched = ops.reshape(
            vision_embeddings,
            [batch_size, total_patches, self.vision_hidden_size],
        )

        transposed = ops.permute(vision_embeddings_batched, [0, 2, 1])

        reshaped = ops.reshape(
            transposed,
            [
                batch_size,
                self.vision_hidden_size,
                self.patches_per_side,
                self.patches_per_side,
            ],
        )

        reshaped = ops.permute(reshaped, [0, 2, 3, 1])

        pooled = avg_pool2d(
            input=reshaped,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=self.kernel_size,
        )

        pooled = ops.permute(pooled, [0, 3, 1, 2])
        pooled = ops.flatten(pooled, start_dim=2, end_dim=3)
        pooled = ops.permute(pooled, [0, 2, 1])

        normalized = self.mm_soft_emb_norm(pooled)

        projection_weight: TensorValue = self.mm_input_projection_weight
        if normalized.device:
            projection_weight = projection_weight.to(normalized.device)

        projection_weight = projection_weight.cast(normalized.dtype)
        projected = normalized @ projection_weight
        projected = projected.cast(self.dtype)
        projected_flat = ops.flatten(projected, start_dim=0, end_dim=1)

        return projected_flat


class Gemma3Multimodal(Module):
    """The Gemma3 multimodal model with vision and language components.

    This is a container class that holds references to the vision encoder,
    multimodal projector, and language model. It is not meant to be called
    directly; instead, use the component models for graph building.

    Components:
    - vision_encoder: SigLipVisionModel for processing images
    - multimodal_projector: Projects vision features to language hidden size
    - language_model: Gemma3 language model with vision embedding support
    """

    def __init__(self, config: Gemma3MultimodalConfig) -> None:
        """Initialize the Gemma3 multimodal model.

        Args:
            config: The Gemma3 multimodal configuration containing parameters
                for vision, text, and projection components.
        """
        super().__init__()
        self.config = config

        # Build vision encoder
        devices = getattr(self.config.text_config, "devices", None)
        self.vision_encoder = SigLipVisionModel(
            self.config.vision_config, devices=devices
        )

        # Build multimodal projector
        device = devices[0] if devices else DeviceRef.CPU()
        self.multimodal_projector = MultimodalProjector(
            vision_hidden_size=self.config.vision_config.hidden_size,
            text_hidden_size=self.config.text_config.hidden_size,
            dtype=self.config.torch_dtype,
            device=device,
            image_size=self.config.vision_config.image_size,
            patch_size=self.config.vision_config.patch_size,
            mm_tokens_per_image=self.config.mm_tokens_per_image,
        )

        # Build language model
        self.language_model = Gemma3LanguageModelWithVision(self.config)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_cache_inputs_per_dev: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
    ) -> tuple[TensorValue, ...]:  # type: ignore[override]
        """This class is not meant to be called directly.

        Use the component models (vision_encoder, language_model) instead for
        graph building, similar to InternVL's architecture pattern.
        """
        raise NotImplementedError(
            "Gemma3Multimodal is a container class. "
            "Use vision_encoder or language_model for graph building."
        )
