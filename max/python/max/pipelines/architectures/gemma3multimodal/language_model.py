# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.abc import Mapping, Sequence

from max.driver import DLPackArray
from max.dtype import DType
from max.graph import BufferValue, TensorValue, ops
from max.graph.weights import WeightData
from max.nn import Module, ReturnLogits
from max.nn.kv_cache import PagedCacheValues
from max.pipelines.architectures.gemma3.gemma3 import Gemma3TextModel
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)

from .model_config import Gemma3MultimodalConfig


class Gemma3LanguageModelWithVision(Module):
    """Gemma3 Language Model wrapper with multimodal embedding support.

    This class wraps the base Gemma3TextModel and adds the ability to
    merge vision embeddings into the text token sequence at specified positions.
    """

    def __init__(self, config: Gemma3MultimodalConfig) -> None:
        super().__init__()
        self.config = config
        # Create the base text model
        self.text_model = Gemma3TextModel(config.text_config)

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue] | None = None,
        image_token_indices: Sequence[TensorValue] | None = None,
    ) -> tuple[TensorValue, ...]:
        """Forward pass with optional image embedding injection.

        Args:
            tokens: Input token IDs.
            signal_buffers: Buffers for device synchronization.
            kv_collections: KV cache collections.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged tensors.
            image_embeddings: Optional vision embeddings to inject.
            image_token_indices: Positions where to inject vision embeddings.

        Returns:
            Model outputs (logits, etc.)
        """
        h = self.text_model.embed_tokens(tokens, signal_buffers)

        if image_embeddings is not None and image_token_indices is not None:
            h = [
                merge_multimodal_embeddings(
                    inputs_embeds=h_device,
                    multimodal_embeddings=img_embed,
                    image_token_indices=img_tok_indices,
                )
                for h_device, img_embed, img_tok_indices in zip(
                    h, image_embeddings, image_token_indices, strict=True
                )
            ]

        for idx, layer in enumerate(self.text_model.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.text_model.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
            )

        last_token_indices = [offsets[1:] - 1 for offsets in input_row_offsets]
        last_token_h = []
        if h:
            last_token_h = [
                ops.gather(h_device, indices, axis=0)
                for h_device, indices in zip(h, last_token_indices, strict=True)
            ]

        last_logits = ops.cast(
            self.text_model.lm_head(
                [
                    self.text_model.norm_shards[i](last_token_h[i])
                    for i in range(len(last_token_h))
                ],
                signal_buffers,
            )[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.text_model.return_logits == ReturnLogits.VARIABLE and h:
            return_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=self.text_model.devices[0],
            )
            last_indices = [
                ops.reshape(
                    ops.unsqueeze(row_offset[1:], -1) - return_range,
                    shape=(-1,),
                )
                for row_offset in input_row_offsets
            ]

            variable_tokens = [
                self.text_model.norm_shards[i](
                    ops.gather(h_device, indices, axis=0)
                )
                for i, (h_device, indices) in enumerate(
                    zip(h, last_indices, strict=True)
                )
            ]
            logits = ops.cast(
                self.text_model.lm_head(variable_tokens, signal_buffers)[0],
                DType.float32,
            )
            offsets = ops.range(
                0,
                last_indices[0].shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.text_model.devices[0],
            )

        elif self.text_model.return_logits == ReturnLogits.ALL and h:
            all_normalized = [
                self.text_model.norm_shards[i](h_device)
                for i, h_device in enumerate(h)
            ]
            logits = ops.cast(
                self.text_model.lm_head(all_normalized, signal_buffers)[0],
                DType.float32,
            )
            offsets = input_row_offsets[0]

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)

        return (last_logits,)

    def load_state_dict(
        self,
        state_dict: Mapping[str, DLPackArray | WeightData],
        *,
        override_quantization_encoding: bool = False,
        weight_alignment: int | None = None,
        strict: bool = True,
    ) -> None:
        """Load weights into the text model."""
        self.text_model.load_state_dict(
            state_dict,
            override_quantization_encoding=override_quantization_encoding,
            weight_alignment=weight_alignment,
            strict=strict,
        )

    def state_dict(
        self, auto_initialize: bool = True
    ) -> dict[str, DLPackArray]:
        """Get the text model's state dict."""
        return self.text_model.state_dict(auto_initialize=auto_initialize)
