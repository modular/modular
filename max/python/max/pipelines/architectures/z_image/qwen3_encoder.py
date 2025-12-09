
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

from max.graph import DeviceRef, TensorValueLike, ops
from max.nn.kv_cache import PagedCacheValues
from max.dtype import DType

from max.pipelines.architectures.qwen3.qwen3 import Qwen3
from max.pipelines.architectures.qwen3.model_config import Qwen3Config

class Qwen3Encoder(Qwen3):
    """
    Qwen3 Encoder wrapper for Z-Image pipeline.
    Accesses the last hidden state of the Qwen3 model to use as text embeddings.
    """
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collection: PagedCacheValues,
        return_n_logits: TensorValueLike,
        input_row_offsets: TensorValueLike,
    ) -> tuple[TensorValueLike, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        # Create position embeddings shared across the decoder layers.
        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        # In Z-Image, we need the hidden states before the LM head.
        # We perform normalization as typical in transformer encoders before output.
        h = self.norm(h)

        # Return hidden states directly. Format: (seq_len, hidden_dim)
        # Note: Depending on usage, we might need to handle batching/offsets if variable length.
        # But for graph output, we just return the tensor.
        return (h,)
