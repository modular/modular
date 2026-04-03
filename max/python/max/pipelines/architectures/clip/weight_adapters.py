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

"""Weight adapters for CLIP vision models."""

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights

CLIP_SAFETENSOR_MAP: dict[str, str] = {
    ".out_proj.": ".o_proj.",
}


def convert_safetensor_state_dict(
    state_dict: Weights,
) -> dict[str, WeightData]:
    new_state_dict: dict[str, WeightData] = {}

    for weight_name, value in state_dict.items():
        max_name = weight_name
        for before, after in CLIP_SAFETENSOR_MAP.items():
            max_name = max_name.replace(before, after)

        weight_data = value.data()
        if max_name == "vision_model.embeddings.patch_embedding.weight":
            assert isinstance(weight_data.data, Buffer)
            if weight_data.dtype == DType.bfloat16:
                data = weight_data.data.view(DType.float16).to_numpy()
            else:
                data = weight_data.data.to_numpy()
            data = data.transpose(2, 3, 1, 0).copy()
            weight_data = WeightData(
                data=data,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=weight_data.shape.__class__(data.shape),
                quantization_encoding=weight_data.quantization_encoding,
            )

        new_state_dict[max_name] = weight_data

    return new_state_dict
