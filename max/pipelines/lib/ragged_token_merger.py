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

from max import nn
from max.dtype import DType
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValue, ops
from max.nn.kernels import merge_ragged_tensors


def ragged_token_merger(device: DeviceRef) -> Graph:
    graph_inputs = [
        TensorType(DType.int64, ["batch_prompt_seq_len"], device=device),
        TensorType(DType.uint32, ["offsets_len"], device=device),
        TensorType(DType.int64, ["batch_size", "draft_seq_len"], device=device),
    ]

    with Graph("merge_prompt_draft_tokens", input_types=graph_inputs) as graph:
        prompt_tensor, prompt_row_offsets, draft_tensor = graph.inputs

        merge_op = RaggedTokenMerger(device)
        merged_tensor, merged_row_offsets = merge_op(
            prompt_tensor.tensor,
            prompt_row_offsets.tensor,
            draft_tensor.tensor,
        )

        graph.output(merged_tensor, merged_row_offsets)

        return graph


class RaggedTokenMerger(nn.Module):
    def __init__(self, device: DeviceRef):
        self.device = device

    def __call__(
        self,
        prompt_tokens: TensorValue,
        prompt_offsets: TensorValue,
        draft_tokens: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        num_steps = ops.cast(
            ops.shape_to_tensor([draft_tokens.shape[1]]).reshape(()),
            DType.uint32,
        )
        draft_tokens_flattened = ops.reshape(draft_tokens, shape=(-1,))
        draft_offsets_len = ops.cast(
            ops.shape_to_tensor([draft_tokens_flattened.shape[0]]).reshape(()),
            DType.uint32,
        )

        draft_offsets = ops.range(
            start=ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
            stop=draft_offsets_len + 1,  # +1 so that we include the end
            step=num_steps,
            out_dim=Dim("offsets_len"),
            device=self.device,
            dtype=DType.uint32,
        )
        merged_tensor, merged_offsets = merge_ragged_tensors(
            prompt_tokens,
            prompt_offsets,
            draft_tokens_flattened,
            draft_offsets,
        )

        return merged_tensor, merged_offsets
