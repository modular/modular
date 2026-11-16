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
"""Assigns Inkling checkpoint tensors to registered role namespaces."""

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import WeightRenaming, WeightRule
from transformers import PretrainedConfig

from ..inkling.weight_adapters import inkling_weight_rules


def inkling_modulev3_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns rules for the registered decoder and vision modules."""
    return (
        WeightRenaming(r"^model\.llm\.", "decoder."),
        *inkling_weight_rules(huggingface_config),
        WeightRenaming(
            r"(\.experts\.w(?:13|2))_weight\.(weight_scale(?:_2)?|input_amax)$",
            r"\1_\2",
        ),
    )
