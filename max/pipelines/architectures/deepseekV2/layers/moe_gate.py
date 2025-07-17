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

"""Mixture of Experts Gate Layer."""

from max.dtype import DType
from max.graph import TensorValue, ops
from max.nn.moe import MoEGate


class DeepSeekV2MoEGate(MoEGate):
    """Mixture of Experts Gate Layer for DeepSeek V2."""

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Compute expert routing weights and indices for input hidden states.

        Args:
            hidden_states: Input tensor of shape (seq_len, hidden_dim)

        Returns:
            tuple containing:
                - topk_idx: Indices of top-k selected experts of shape (seq_len, num_experts_per_token)
                - topk_weight: Routing weights for selected experts of shape (seq_len, num_experts_per_token)
        """
        # compute gating score
        logits = self.gate_score(hidden_states.cast(DType.float32))
        scores = ops.softmax(logits.cast(DType.float32))

        # select top k experts
        topk_weight, topk_idx = ops.top_k(
            scores, self.num_experts_per_token, -1
        )

        return topk_idx, topk_weight
