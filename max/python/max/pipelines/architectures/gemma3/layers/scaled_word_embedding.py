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

"""Scaled Word Embedding."""

from collections.abc import Iterable

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, TensorValueLike, ops
from max.graph.quantization import QuantizationEncoding
from max.nn.embedding import VocabParallelEmbedding


class ScaledWordEmbedding(VocabParallelEmbedding):
    """
    This layer is a wrapper around nn.Embedding that multiplies the embeddings
    with a scale factor.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        dtype: DType,
        devices: list[DeviceRef],
        quantization_encoding: QuantizationEncoding | None = None,
        name: str | None = None,
        embed_scale: float = 1.0,
    ) -> None:
        """Initializes the embedding layer with the given arguments.

        Args:
            vocab_size: The number of unique items in the vocabulary.
                Indices must be in the range ``[0, vocab_size)``.
            hidden_dim: The dimensionality of each embedding vector.
            dtype: The data type of the embedding weights.
            device: The device where embedding lookups are executed.
                Model init transfers the initially CPU-resident weights to this
                device.
            quantization_encoding: The quantization encoding to use for the
                embedding weights.
            name: The name identifier for the embedding weight matrix.
            embed_scale: The scale to multiply the embeddings with.
        """
        super().__init__(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            dtype=dtype,
            devices=devices,
            quantization_encoding=quantization_encoding,
            name=name,
        )
        self.embed_scale = embed_scale

    def __call__(
        self, indices: TensorValueLike, signal_buffers: Iterable[BufferValue]
    ) -> list[TensorValue]:
        """Embeds the input indices by looking up corresponding vectors, then
        multiplies the embeddings with the scale factor.

        Args:
            indices: A tensor of integer indices to look up.
                Each index must be in the range ``[0, vocab_size)``.

        Returns:
            A tensor containing the embeddings corresponding to the input
            indices, multiplied by the scale factor.
            The result resides on the device specified in :obj:`device`.
        """
        result = super().__call__(indices, signal_buffers)
        return [
            result_device
            * ops.constant(
                self.embed_scale,
                result_device.dtype,
                device=result_device.type.device,
            )
            for result_device in result
        ]
