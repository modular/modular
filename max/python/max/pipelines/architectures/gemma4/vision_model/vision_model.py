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

from __future__ import annotations

from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    TensorType,
    TensorValue,
    Weight,
)
from max.nn.layer import Module
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    compute_vision_freqs_cis,
)

from ..layers.multimodal_embedder import Gemma4MultimodalEmbedder
from ..model_config import Gemma4ForConditionalGenerationConfig
from .embedding import Gemma4VisionPatchEmbedder
from .encoding import Gemma4VisionEncoder
from .pooling import Gemma4VisionPooler


class Gemma4VisionModel(Module):
    """Vision tower for the Gemma4 multimodal model.

    Processes a ragged batch of flat image patches through:

    1. ``patch_embedder`` — pixel normalisation, linear projection, 2-D
       position embeddings.
    2. ``encoder`` — stack of ``Gemma4VisionEncoderLayer`` blocks with 2-D
       multidimensional RoPE and ragged flash attention.
    3. ``pooler`` — sparse average pooling to ``image_seq_length`` tokens
       per image using a pre-computed weight matrix passed as a graph input.
    """

    def __init__(
        self, config: Gemma4ForConditionalGenerationConfig, device: DeviceRef
    ) -> None:
        if len(config.devices) > 1:
            raise ValueError("Gemma4VisionModel only supports a single device")

        super().__init__()
        self.config = config
        self.device = config.devices[0]
        vision_config = config.vision_config
        self.patch_embedder = Gemma4VisionPatchEmbedder(
            config, device=self.device
        )
        self.encoder = Gemma4VisionEncoder(config)
        self.pooler = Gemma4VisionPooler(vision_config.hidden_size)
        self.rope_theta = vision_config.rope_theta
        self.head_dim = vision_config.head_dim

        # In the reference, the multimodal embedder is in the layer above
        # (Gemma4Model), but the image features includes this projection,
        # so it is included here.
        vision_dtype = config.unquantized_dtype
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_config.hidden_size,
            config.text_config.hidden_size,
            vision_dtype,
            self.device,
            vision_config.rms_norm_eps,
        )
        self.standardize = vision_config.standardize

        if self.standardize:
            self.std_bias = Weight(
                "std_bias",
                vision_dtype,
                shape=[vision_config.hidden_size],
                device=self.device,
            )
            self.std_scale = Weight(
                "std_scale",
                vision_dtype,
                shape=[vision_config.hidden_size],
                device=self.device,
            )

    def __call__(
        self,
        patches_flat: TensorValue,
        pixel_position_ids: TensorValue,
        cu_seqlens: TensorValue,
        pool_weights: TensorValue,
        max_seq_len: TensorValue,
    ) -> TensorValue:
        """Process packed image patches through the full vision tower.

        Args:
            patches_flat: Packed flat patch pixels,
                shape ``[total_patches, 3 * patch_size²]``.
            pixel_position_ids: Integer (x, y) coordinates,
                shape ``[total_patches, 2]``, dtype int32.
            cu_seqlens: Cumulative sequence lengths
                (image boundaries), shape ``[num_images + 1]``, dtype uint32.
            pool_weights: Sparse pooling weight matrices,
                shape ``[num_images * image_seq_length, total_patches]``,
                dtype bfloat16.
            max_seq_len: Maximum patches per image (scalar uint32, CPU).

        Returns:
            Projected image embeddings, shape
            ``[num_images * image_seq_length, text_hidden_size]``.
        """
        hidden_states = self.patch_embedder(patches_flat, pixel_position_ids)

        freqs_cis = compute_vision_freqs_cis(
            pixel_position_ids=pixel_position_ids,
            head_dim=self.head_dim,
            ndim=2,
            theta=self.rope_theta,
            dtype=DType.bfloat16,
            device=self.device,
        )
        encoded = self.encoder(
            hidden_states, freqs_cis, cu_seqlens, max_seq_len
        )

        pooled = self.pooler(encoded, pool_weights)

        if self.standardize:
            pooled = (pooled - self.std_bias) * self.std_scale

        return self.embed_vision(pooled)

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Build the input type list for the vision model graph.

        The vision model receives five device tensors plus one CPU scalar.

        * ``patches_flat``        — ``[total_patches, 3 * patch_size²]``, bf16
        * ``pixel_position_ids``  — ``[total_patches, 2]``, int32
        * ``cu_seqlens``          — ``[num_images + 1]``, uint32
        * ``pool_weights``        — ``[num_pooled_tokens, total_patches]``, bf16
        * ``max_seq_len``         — scalar uint32, on CPU (one shared tensor)
        """
        vision_config = self.config.vision_config
        patch_dim = 3 * vision_config.patch_size**2

        patches_flat_type = TensorType(
            DType.bfloat16,
            shape=["total_patches", patch_dim],
            device=self.device,
        )
        pixel_position_ids_type = TensorType(
            DType.int32,
            shape=["total_patches", 2],
            device=self.device,
        )
        cu_seqlens_type = TensorType(
            DType.uint32,
            shape=["num_images_plus_1"],
            device=self.device,
        )
        pool_weights_type = TensorType(
            DType.float32,
            shape=["num_pooled_tokens", "total_patches"],
            device=self.device,
        )
        max_seq_len_type = TensorType(
            DType.uint32,
            shape=[],
            device=DeviceRef.CPU(),
        )
        return (
            patches_flat_type,
            pixel_position_ids_type,
            cu_seqlens_type,
            pool_weights_type,
            max_seq_len_type,
        )
