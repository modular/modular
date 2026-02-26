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

import torch
from max.driver import CPU, Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import merge_ragged_tensors

BATCH_SIZE = 5
HIDDEN_DIM = 64


def torch_merge_ragged_tensors(
    a_input: torch.Tensor,
    a_batch_sizes: torch.Tensor,
    a_offsets: torch.Tensor,
    b_input: torch.Tensor,
    b_batch_sizes: torch.Tensor,
    b_offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch_merged_tensor = torch.zeros(
        (
            int(torch.sum(a_batch_sizes).item())
            + int(torch.sum(b_batch_sizes).item()),
            *a_input.shape[1:],
        ),
        dtype=torch.int32,
        device="cpu",
    )

    row_idx = 0
    for i in range(BATCH_SIZE):
        torch_merged_tensor[row_idx : row_idx + a_batch_sizes[i]] = a_input[
            a_offsets[i] : a_offsets[i + 1]
        ]
        row_idx += a_batch_sizes[i]
        torch_merged_tensor[row_idx : row_idx + b_batch_sizes[i]] = b_input[
            b_offsets[i] : b_offsets[i + 1]
        ]
        row_idx += b_batch_sizes[i]

    torch_merged_row_offsets = torch.zeros(
        (BATCH_SIZE + 1,),
        dtype=torch.uint32,
        device="cpu",
    )
    torch_merged_row_offsets[1:] = torch.cumsum(
        a_batch_sizes + b_batch_sizes, dim=0
    )

    return torch_merged_tensor, torch_merged_row_offsets


def merge_ragged_tensors_1d(device: Device, device_ref: DeviceRef) -> None:
    a_batch_sizes = torch.randint(1, 20, (BATCH_SIZE,))
    # a_row_offsets = torch.cumsum(a_batch_sizes, dim=0), with the first element being 0
    a_offsets = torch.zeros(BATCH_SIZE + 1, dtype=torch.uint32, device="cpu")
    a_offsets[1:] = torch.cumsum(a_batch_sizes, dim=0)
    a_input = torch.randint(
        0,
        100000,
        (int(torch.sum(a_batch_sizes).item()),),
        dtype=torch.int32,
        device="cpu",
    )

    b_batch_sizes = torch.randint(1, 20, (BATCH_SIZE,))
    b_offsets = torch.zeros(BATCH_SIZE + 1, dtype=torch.uint32, device="cpu")
    b_offsets[1:] = torch.cumsum(b_batch_sizes, dim=0)
    b_input = torch.randint(
        0,
        100000,
        (int(torch.sum(b_batch_sizes).item()),),
        dtype=torch.int32,
        device="cpu",
    )

    # Construct input types.
    a_input_type = TensorType(
        DType.int32,
        ["a_seq_len"],
        device=device_ref,
    )
    a_offsets_type = TensorType(
        DType.uint32,
        ["offsets_len"],
        device=device_ref,
    )
    b_input_type = TensorType(
        DType.int32,
        ["b_seq_len"],
        device=device_ref,
    )
    b_offsets_type = TensorType(
        DType.uint32,
        ["offsets_len"],
        device=device_ref,
    )

    with Graph(
        "merge_ragged_tensors_1d",
        input_types=(
            a_input_type,
            a_offsets_type,
            b_input_type,
            b_offsets_type,
        ),
    ) as graph:
        a_tensor, a_row_offsets, b_tensor, b_row_offsets = graph.inputs
        merged_tensor, merged_row_offsets = merge_ragged_tensors(
            a_tensor.tensor,
            a_row_offsets.tensor,
            b_tensor.tensor,
            b_row_offsets.tensor,
        )

        graph.output(merged_tensor, merged_row_offsets)

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    results = compiled.execute(
        Buffer.from_dlpack(a_input).to(device),
        Buffer.from_dlpack(a_offsets).to(device),
        Buffer.from_dlpack(b_input).to(device),
        Buffer.from_dlpack(b_offsets).to(device),
    )

    max_merged_tensor = torch.from_dlpack(results[0]).to("cpu")
    max_merged_row_offsets = torch.from_dlpack(results[1]).to("cpu")

    torch_merged_tensor, torch_merged_row_offsets = torch_merge_ragged_tensors(
        a_input, a_batch_sizes, a_offsets, b_input, b_batch_sizes, b_offsets
    )

    assert torch.all(max_merged_tensor == torch_merged_tensor)
    assert torch.all(max_merged_row_offsets == torch_merged_row_offsets)


def merge_ragged_tensors_2d(device: Device, device_ref: DeviceRef) -> None:
    a_batch_sizes = torch.randint(1, 20, (BATCH_SIZE,))
    # a_row_offsets = torch.cumsum(a_batch_sizes, dim=0), with the first element being 0
    a_offsets = torch.zeros(BATCH_SIZE + 1, dtype=torch.uint32, device="cpu")
    a_offsets[1:] = torch.cumsum(a_batch_sizes, dim=0)
    a_input = torch.randint(
        0,
        100000,
        (int(torch.sum(a_batch_sizes).item()), HIDDEN_DIM),
        dtype=torch.int32,
        device="cpu",
    )

    b_batch_sizes = torch.randint(1, 20, (BATCH_SIZE,))
    b_offsets = torch.zeros(BATCH_SIZE + 1, dtype=torch.uint32, device="cpu")
    b_offsets[1:] = torch.cumsum(b_batch_sizes, dim=0)
    b_input = torch.randint(
        0,
        100000,
        (int(torch.sum(b_batch_sizes).item()), HIDDEN_DIM),
        dtype=torch.int32,
        device="cpu",
    )

    # Construct input types.
    a_input_type = TensorType(
        DType.int32,
        ["a_seq_len", HIDDEN_DIM],
        device=device_ref,
    )
    a_offsets_type = TensorType(
        DType.uint32,
        ["offsets_len"],
        device=device_ref,
    )
    b_input_type = TensorType(
        DType.int32,
        ["b_seq_len", HIDDEN_DIM],
        device=device_ref,
    )
    b_offsets_type = TensorType(
        DType.uint32,
        ["offsets_len"],
        device=device_ref,
    )

    with Graph(
        "merge_ragged_tensors_1d",
        input_types=(
            a_input_type,
            a_offsets_type,
            b_input_type,
            b_offsets_type,
        ),
    ) as graph:
        a_tensor, a_row_offsets, b_tensor, b_row_offsets = graph.inputs
        merged_tensor, merged_row_offsets = merge_ragged_tensors(
            a_tensor.tensor,
            a_row_offsets.tensor,
            b_tensor.tensor,
            b_row_offsets.tensor,
        )

        graph.output(merged_tensor, merged_row_offsets)

    session = InferenceSession(devices=[device])
    compiled = session.load(graph)

    results = compiled.execute(
        Buffer.from_dlpack(a_input).to(device),
        Buffer.from_dlpack(a_offsets).to(device),
        Buffer.from_dlpack(b_input).to(device),
        Buffer.from_dlpack(b_offsets).to(device),
    )

    max_merged_tensor = torch.from_dlpack(results[0]).to("cpu")
    max_merged_row_offsets = torch.from_dlpack(results[1]).to("cpu")

    torch_merged_tensor, torch_merged_row_offsets = torch_merge_ragged_tensors(
        a_input, a_batch_sizes, a_offsets, b_input, b_batch_sizes, b_offsets
    )

    assert torch.all(max_merged_tensor == torch_merged_tensor)
    assert torch.all(max_merged_row_offsets == torch_merged_row_offsets)


def test_merge_ragged_tensors_1d() -> None:
    merge_ragged_tensors_1d(Accelerator(0), DeviceRef.GPU())
    merge_ragged_tensors_1d(CPU(0), DeviceRef.CPU())


def test_merge_ragged_tensors_2d() -> None:
    merge_ragged_tensors_2d(Accelerator(0), DeviceRef.GPU())
    merge_ragged_tensors_2d(CPU(0), DeviceRef.CPU())
