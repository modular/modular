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

"""Weight conversion utilities for Qwen3VL tests."""

import torch
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
    Qwen3VLMoeVisionPatchMerger as HFQwen3VLMoeVisionPatchMerger,
)


def load_weights_to_hf_merger(
    hf_merger: HFQwen3VLMoeVisionPatchMerger,
    weights: dict[str, torch.Tensor],
) -> None:
    """Load weights into HuggingFace merger."""
    device = weights["norm.weight"].device
    # Load norm weights
    hf_merger.norm.weight.data = weights["norm.weight"].cuda()
    hf_merger.norm.bias.data = weights["norm.bias"].cuda()

    # Load MLP weights
    hf_merger.linear_fc1.weight.data = weights["linear_fc1.weight"].cuda()
    hf_merger.linear_fc1.bias.data = weights["linear_fc1.bias"].cuda()
    hf_merger.linear_fc2.weight.data = weights["linear_fc2.weight"].cuda()
    hf_merger.linear_fc2.bias.data = weights["linear_fc2.bias"].cuda()
