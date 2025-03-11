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

from max.graph import TensorValue
from max.nn.layer import Module
from max.pipelines.architectures.deepseekV2.layers.yarn_rope import (
    YarnRotaryEmbedding,
)


class DeepseekAttention(Module):
    def __init__(self, layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx

        # TODO: Add DeepseekConfig for these parameters once it lands (MODELS-369).
        self.attention_dropout = 0.0
        self.hidden_size = 4096
        self.num_heads = 32

        self.max_position_embeddings = 2048
        self.rope_theta = 10000.0
        self.q_lora_rank = 1536
        self.qk_rope_head_dim = 64
        self.kv_lora_rank = 512
        self.v_head_dim = 128
        self.qk_nope_head_dim = 128
        self.rope_theta = 10000.0
        self.q_head_dim = 128 + 64
        self.is_causal = True

        self.rotary_emb = YarnRotaryEmbedding(
            self.qk_rope_head_dim,
        )

    def __call__(self, x: TensorValue, attention_mask: TensorValue):
        seq_len = x.shape[2]

        # TODO: Call new MLA kernel once it lands (E2EOPT-44).
        raise NotImplementedError("DeepseekAttention is not implemented")
