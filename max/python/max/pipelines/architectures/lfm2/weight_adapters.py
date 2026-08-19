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

from max.graph.weights import WeightData, Weights

LFM2_SAFETENSOR_MAPPING = (
    ("model.embed_tokens", "embed_tokens"),
    ("model.embedding_norm", "norm"),
    ("model.layers", "layers"),
    ("self_attn.out_proj", "self_attn.o_proj"),
    ("self_attn.q_layernorm", "self_attn.q_norm"),
    ("self_attn.k_layernorm", "self_attn.k_norm"),
    ("conv.conv.weight", "conv.conv_weight"),
    ("conv.conv.bias", "conv.conv_bias"),
)


def convert_lfm2_safetensor_state_dict(
    state_dict: dict[str, Weights],
    **unused_kwargs,
) -> dict[str, WeightData]:
    mapped: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        max_name = name
        for before, after in LFM2_SAFETENSOR_MAPPING:
            max_name = max_name.replace(before, after)
        mapped[max_name] = value.data()
    return mapped
