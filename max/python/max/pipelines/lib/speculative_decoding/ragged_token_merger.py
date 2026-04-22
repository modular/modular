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

"""Implements ragged token merging for speculative decoding workflows."""

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValue, ops
from max.nn.kernels import merge_ragged_tensors
from max.nn.layer import Module


def shape_to_scalar(
    x: Dim, device: DeviceRef, dtype: DType = DType.int64
) -> TensorValue:
    """An especially cursed way to turn a shape into a scalar on the GPU.

    Usage:
    >>> my_scalar = shape_to_scalar(tensor.shape[3], DeviceRef.GPU())

    We previously did `ops.shape_to_tensor([x]).to(DeviceRef.GPU())` but this
    issues a h2d copy that is incompatible with CUDA Graphs.

    TODO: Delete this method asap! We should really support a device placement
    when using shape_to_tensor.
    """
    x_cpu = ops.shape_to_tensor([x])
    x_gpu = ops.range(
        start=x_cpu[0],
        stop=x_cpu[0] + 1,
        out_dim=1,
        dtype=dtype,
        device=device,
    )[0]
    return x_gpu


def ragged_token_merger(device: DeviceRef) -> Graph:
    """Builds a graph that merges prompt and draft tokens into a single ragged sequence.

    Args:
        device: Device for the graph inputs and merge op.

    Returns:
        A graph that takes prompt tokens, prompt row offsets, and draft tokens and
        outputs merged tokens and merged row offsets.
    """
    graph_inputs = [
        TensorType(DType.int64, ["batch_prompt_seq_len"], device=device),
        TensorType(DType.uint32, ["offsets_len"], device=device),
        TensorType(DType.int64, ["batch_size", "draft_seq_len"], device=device),
    ]

    with Graph("merge_prompt_draft_tokens", input_types=graph_inputs) as graph:
        prompt_tensor, prompt_row_offsets, draft_tensor = graph.inputs

        merge_op = RaggedTokenMerger(device)
        merged_tensor, merged_row_offsets = merge_op(
            prompt_tensor.tensor, prompt_row_offsets.tensor, draft_tensor.tensor
        )

        graph.output(merged_tensor, merged_row_offsets)

        return graph


class RaggedTokenMerger(Module):
    """Merges prompt and draft token sequences into a single ragged batch."""

    def __init__(self, device: DeviceRef) -> None:
        super().__init__()
        self.device = device

    def __call__(
        self,
        prompt_tokens: TensorValue,
        prompt_offsets: TensorValue,
        draft_tokens: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Merges prompt and draft tokens into a single ragged token sequence.

        Args:
            prompt_tokens: The prompt tokens of shape [S].
            prompt_offsets: The prompt offsets of shape [B+1].
            draft_tokens: The draft tokens of shape [B, K].

        Returns:
            A tuple of two tensors:
                - The merged tokens of shape [S+B*K].
                - The merged offsets of shape [B+1].
        """
        device = prompt_tokens.device
        K = shape_to_scalar(draft_tokens.shape[1], device, dtype=DType.uint32)
        draft_tokens_flattened = ops.reshape(draft_tokens, shape=(-1,))

        # Compute draft_offsets as [0, K, 2K, ..., N*K] where K=num_steps.
        # We use range(step=1) * num_steps instead of range(step=num_steps)
        # because step=0 (during prefill when draft_seq_len=0) is undefined.
        batch_size_plus_1 = ops.shape_to_tensor([prompt_offsets.shape[0]])[0]
        indices = ops.range(
            start=0,
            stop=batch_size_plus_1,
            out_dim=prompt_offsets.shape[0],
            device=device,
            dtype=DType.uint32,
        )
        draft_offsets = indices * K
        merged_tensor, merged_offsets = merge_ragged_tensors(
            prompt_tokens, prompt_offsets, draft_tokens_flattened, draft_offsets
        )

        return merged_tensor, merged_offsets


class RaggedTokenMergerRunner:
    """Runner for the ragged token merger."""

    def __init__(
        self, session: InferenceSession, device_ref: DeviceRef
    ) -> None:
        self._model = session.load(ragged_token_merger(device=device_ref))
        self._device = device_ref.to_device()

    def run(
        self,
        tokens: Buffer,
        input_row_offsets: Buffer,
        draft_tokens: Buffer,
    ) -> tuple[Buffer, Buffer]:
        """Runs the ragged token merger."""
        if tokens.device != self._device:
            raise ValueError(
                f"Tokens must be on device {self._device}, got {tokens.device}"
            )
        if input_row_offsets.device != self._device:
            raise ValueError(
                f"Input row offsets must be on device {self._device}, got {input_row_offsets.device}"
            )
        if draft_tokens.device != self._device:
            raise ValueError(
                f"Draft tokens must be on device {self._device}, got {draft_tokens.device}"
            )
        merged_tokens, merged_offsets = self._model(
            tokens, input_row_offsets, draft_tokens
        )
        return merged_tokens, merged_offsets
