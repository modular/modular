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
from max.graph.weights import WeightData, Weights


def _prepare_state_dict(
    weights: Weights,
    target_dtype: DType | None = None,
) -> dict[str, WeightData]:
    """Convert Weights to a raw state dict, normalizing tied embedding keys.

    HF UMT5 ties ``shared.weight`` and ``encoder.embed_tokens.weight``.
    Our module owns the embedding as ``shared``, so we normalize to that key
    and drop the alias to avoid strict-mode validation failures.

    If ``target_dtype`` is provided, all weights are cast to that dtype
    (e.g. float32 → bfloat16 for Wan 2.1 checkpoints).
    """
    state_dict: dict[str, WeightData] = {}
    for key, value in weights.items():
        wd = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        state_dict[key] = wd

    encoder_emb = state_dict.pop("encoder.embed_tokens.weight", None)
    if "shared.weight" not in state_dict and encoder_emb is not None:
        state_dict["shared.weight"] = encoder_emb

    return state_dict
