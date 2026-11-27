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
"""Names Inkling checkpoint tensors as the decoder and vision modules load them."""

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import (
    Ignore,
    WeightRenaming,
    WeightRule,
)
from transformers import PretrainedConfig


def _checkpoint_rules() -> Sequence[WeightRule]:
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
        WeightRenaming(r"^model\.visual\.", "vision."),
        WeightRenaming(r"\.scale2$", ".weight_scale_2"),
        WeightRenaming(r"\.scale$", ".weight_scale"),
    )


def inkling_modulev3_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns rules for the registered decoder and vision modules."""
    del huggingface_config
    return (
        WeightRenaming(r"^model\.llm\.", "decoder."),
        *_checkpoint_rules(),
        WeightRenaming(
            r"(\.experts\.w(?:13|2))_weight\.(weight_scale(?:_2)?|input_amax)$",
            r"\1_\2",
        ),
    )
