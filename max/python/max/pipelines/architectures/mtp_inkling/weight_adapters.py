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
"""Names the MTP tensors stored alongside an Inkling target checkpoint."""

from collections.abc import Sequence

from max.pipelines.lib.weight_transform import (
    Ignore,
    WeightRenaming,
    WeightRule,
)
from transformers import PretrainedConfig


def inkling_mtp_weight_rules(
    huggingface_config: PretrainedConfig,
) -> Sequence[WeightRule]:
    """Returns the rules for the checkpoint's MTP depths."""
    del huggingface_config
    return (
        Ignore(r"\.original_shape$", reason="NVFP4 original-shape metadata"),
        Ignore(
            r"^model\.(?:llm|audio|visual)\.",
            reason="the draft loads only its MTP depths",
        ),
        WeightRenaming(r"^model\.mtp\.", ""),
        WeightRenaming(
            r"^(layers\.\d+)\.transformer_block\.", r"\1.decoder_layer."
        ),
    )
