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
"""Tests the RandomWeight class."""

import torch
from max.dtype import DType
from max.graph.weights import RandomWeights


def test_random_weight() -> None:
    """Tests that random weight creation works, checking shape and dtype."""
    weights = RandomWeights()
    _ = weights.vision_model.gated_positional_embedding.gate.allocate(
        DType.bfloat16, [1]
    )
    materialized_weights = weights.allocated_weights
    weight_name = "vision_model.gated_positional_embedding.gate"
    assert weight_name in materialized_weights
    assert materialized_weights[weight_name].dtype == torch.bfloat16
