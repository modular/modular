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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.conv import Conv2d
from max.nn.embedding import Embedding
from max.nn.layer import Module

from ..model_config import ClipVisionConfig


class CLIPVisionEmbeddings(Module):
    def __init__(
        self,
        config: ClipVisionConfig,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.class_embedding = Weight(
            "class_embedding",
            dtype,
            [self.embed_dim],
            device,
        )
        self.patch_embedding = Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            has_bias=False,
            dtype=dtype,
            device=device,
        )
        self.position_embedding = Embedding(
            vocab_size=self.num_positions,
            hidden_dim=self.embed_dim,
            dtype=dtype,
            device=device,
        )
        self.device = device

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        batch_size = pixel_values.shape[0]

        pixel_values_nhwc = ops.permute(pixel_values, [0, 2, 3, 1])
        patch_embeds = self.patch_embedding(pixel_values_nhwc)
        patch_embeds = ops.reshape(
            patch_embeds,
            [batch_size, self.num_patches, self.embed_dim],
        )

        cls_token = ops.reshape(self.class_embedding, [1, 1, self.embed_dim])
        cls_tokens = ops.broadcast_to(
            cls_token,
            [batch_size, 1, self.embed_dim],
        )
        embeddings = ops.concat([cls_tokens, patch_embeds], axis=1)

        position_ids = ops.range(
            start=0,
            stop=self.num_positions,
            step=1,
            out_dim=self.num_positions,
            device=self.device,
            dtype=DType.int32,
        )
        position_ids = ops.unsqueeze(position_ids, 0)
        position_ids = ops.broadcast_to(
            position_ids,
            [batch_size, self.num_positions],
        )
        embeddings = embeddings + self.position_embedding(position_ids)
        return embeddings
