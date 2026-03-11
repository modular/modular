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

from max.experimental.nn.common_layers.attention import AttentionWithRope
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.conv import Conv2d
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import RMSNorm

from ..llama3_modulev3.layers.transformer_block import LlamaTransformerBlock
from .llava.llava import LlavaConditionalGeneration
from .llava.llava_decoder import Transformer
from .llava.llava_projector import LlavaMultiModalConnector
from .model_config import PixtralConfig
from .vision_encoder.attention import Attention as VisionEncoderAttention
from .vision_encoder.rotary_embedding_2d import RotaryEmbedding2D
from .vision_encoder.transformer import MLP as VisionEncoderMLP
from .vision_encoder.transformer import Transformer as VisionEncoderTransformer
from .vision_encoder.transformer import TransformerBlock as VisionEncoderBlock
from .vision_encoder.vision_encoder import VisionEncoder


class Pixtral(LlavaConditionalGeneration):
    """The overall interface to the Pixtral model."""

    def __init__(self, config: PixtralConfig) -> None:
        vision_encoder: VisionEncoder = self._vision_encoder(config)
        multi_modal_projector: LlavaMultiModalConnector = (
            LlavaMultiModalConnector(
                hidden_size=config.hidden_size,
                vision_hidden_size=config.vision_hidden_size,
            )
        )

        language_model: Transformer = self._language_model(config)
        image_token_index: int = config.image_token_index

        super().__init__(
            vision_encoder=vision_encoder,
            multi_modal_projector=multi_modal_projector,
            language_model=language_model,
            image_token_index=image_token_index,
            kv_params=config.kv_params,
        )

    def _language_model(self, config: PixtralConfig) -> Transformer:
        # copied from mistral.py
        # hidden_size (5120) != num_attention_heads * head_dim (128 * 40 = 4,096)
        rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.devices[0].to_device(),
            interleaved=False,
        )

        layers = [
            LlamaTransformerBlock(
                attention=AttentionWithRope(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    scale=config.attention_multiplier,
                    stacked_qkv=False,
                    has_bias=False,
                ),
                mlp=MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.feed_forward_length,
                ),
                input_layernorm=RMSNorm(
                    dim=config.hidden_size,
                    eps=config.rms_norm_eps,
                ),
                post_attention_layernorm=RMSNorm(
                    dim=config.hidden_size,
                    eps=config.rms_norm_eps,
                ),
            )
            for i in range(config.num_hidden_layers)
        ]

        # Create Embedding and output layers.
        embedding_layer = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )
        output = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        return Transformer(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=RMSNorm(
                dim=config.hidden_size,
                eps=config.rms_norm_eps,
            ),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            return_logits=config.return_logits,
        )

    def _vision_encoder(self, config: PixtralConfig) -> VisionEncoder:
        """Creates a 2D convolution layer with the following assumptions:
        - kernel size = (patch_size, patch_size)
        - stride = (patch_size, patch_size)
        - padding = (0, 0, 0, 0)

        This convolution splits the image into patches and then learns an embedding
        of each patch. The embedding dim is out_channels.
        """
        patch_conv = Conv2d(
            permute=True,
            kernel_size=(config.patch_size, config.patch_size),
            in_channels=config.num_channels,
            out_channels=config.vision_hidden_size,
            stride=(config.patch_size, config.patch_size),
            has_bias=False,
        )
        ln_pre = RMSNorm(
            dim=config.vision_hidden_size,
            eps=1e-5,
        )
        patch_rope = RotaryEmbedding2D(
            dim=config.vision_hidden_size,
            n_heads=config.vision_num_attention_heads,
            theta=config.vision_rope_theta,
            max_patches_per_side=config.image_size // config.patch_size,
        )
        encoder_transformer = VisionEncoderTransformer(
            n_heads=config.vision_num_attention_heads,
            layers=[
                VisionEncoderBlock(
                    attention=VisionEncoderAttention(
                        n_heads=config.vision_num_attention_heads,
                        dim=config.vision_hidden_size,
                        head_dim=config.vision_head_dim,
                    ),
                    feed_forward=VisionEncoderMLP(
                        hidden_size=config.vision_hidden_size,
                        intermediate_size=config.vision_intermediate_size,
                    ),
                    attention_norm=RMSNorm(
                        dim=config.vision_hidden_size,
                        eps=1e-5,
                    ),
                    ffn_norm=RMSNorm(
                        dim=config.vision_hidden_size,
                        eps=1e-5,
                    ),
                )
                for i in range(config.vision_num_hidden_layers)
            ],
            dtype=config.dtype,
        )
        return VisionEncoder(
            patch_conv=patch_conv,
            layer_norm=ln_pre,
            patch_positional_embedding=patch_rope,
            transformer=encoder_transformer,
            dtype=config.dtype,
            patch_size=config.patch_size,
            hidden_size=config.vision_hidden_size,
            max_image_size=config.image_size,
        )
