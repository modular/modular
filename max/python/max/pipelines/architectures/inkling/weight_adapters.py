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
"""Names Inkling checkpoint tensors as the decoder and vision graph load them."""

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import (
    Ignore,
    WeightRenaming,
    WeightRule,
)
from transformers import PretrainedConfig

_TEXT = "model.llm."
_VISION = "model.visual."
VISION_PREFIX = "vision."


def inkling_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules for Inkling's text and vision checkpoint tensors."""
    del huggingface_config
    return (
        Ignore(r"\.original_shape$", reason="NVFP4 original-shape metadata"),
        Ignore(r"^model\.mtp\.", reason="MTP depths load as their own draft"),
        Ignore(r"^model\.audio\.", reason="the audio tower is not served"),
        WeightRenaming(r"^model\.llm\.", ""),
        WeightRenaming(
            r"^model\.visual\.layers\.linear_(\d+)\.", r"vision.linears.\1."
        ),
        WeightRenaming(
            r"^model\.visual\.layers\.norm_(\d+)\.", r"vision.norms.\1."
        ),
        WeightRenaming(r"^model\.visual\.", VISION_PREFIX),
        WeightRenaming(r"\.scale2$", ".weight_scale_2"),
        WeightRenaming(r"\.scale$", ".weight_scale"),
    )
