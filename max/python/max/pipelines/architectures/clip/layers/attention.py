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
from max.graph import DeviceRef, TensorValue
from max.nn.attention.multihead_attention import MultiheadAttention


class CLIPVisionAttention(MultiheadAttention):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={hidden_size}, "
                f"num_attention_heads={num_attention_heads})."
            )

        self.embed_dim = hidden_size
        self.o_proj_has_bias = True
        super().__init__(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            devices=[device],
            dtype=dtype,
            scale=(hidden_size // num_attention_heads) ** (-0.5),
            qkv_has_bias=True,
            o_proj_has_bias=self.o_proj_has_bias,
            stacked_qkv=False,
        )

    def __call__(self, x: TensorValue, **kwargs: object) -> TensorValue:
        q, k, v = self._compute_qkv(x)
        attn_out = self._apply_attention(q, k, v, **kwargs)
        return self.o_proj(attn_out)
