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
"""Unit tests for LoRA SGMV kernels with PyTorch reference implementations."""

import numpy as np
import numpy.typing as npt
import pytest
import torch
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.kernels import (
    sgmv_kernel,
)
from torch.utils.dlpack import from_dlpack

DTYPE = DType.bfloat16
TORCH_DTYPE = torch.bfloat16
RTOL = 1e-3
ATOL = 5e-3


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------


def rand_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype = TORCH_DTYPE,
    seed: int = 1234,
) -> torch.Tensor:
    """Generate random tensor on CPU."""
    torch.manual_seed(seed)
    return torch.randn(shape, dtype=dtype) * 0.02


def create_session() -> tuple[InferenceSession, DeviceRef, Device]:
    device = Accelerator(0)
    session = InferenceSession(devices=[device])
    return session, DeviceRef.from_device(device), device


def create_groups(
    total_tokens: int, num_groups: int
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    """Create evenly-divided group offsets and adapter IDs (all using adapter 0)."""
    tokens_per_group = total_tokens // num_groups
    offsets = np.array(
        [i * tokens_per_group for i in range(num_groups)] + [total_tokens],
        dtype=np.uint32,
    )
    lora_ids = np.zeros(num_groups, dtype=np.int32)
    return offsets, lora_ids


def to_max_tensor(tensor: torch.Tensor, device: Device) -> Buffer:
    """Convert CPU torch tensor to MAX tensor."""
    return Buffer.from_dlpack(tensor).to(device)


def to_torch(tensor: Buffer) -> torch.Tensor:
    """Convert MAX tensor to torch tensor."""
    return from_dlpack(tensor)


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    """Compare tensors, moving to CPU for comparison."""
    torch.testing.assert_close(
        actual.cpu(), expected.cpu(), rtol=rtol, atol=atol
    )


def calc_input_row_offsets(seq_lens: list[int]) -> npt.NDArray[np.uint32]:
    """Calculate cumulative input row offsets from sequence lengths."""
    return np.cumsum([0] + seq_lens, dtype=np.uint32)


# -----------------------------------------------------------------------------
# PyTorch Reference Implementations
# -----------------------------------------------------------------------------


def torch_sgmv_kernel(
    input: torch.Tensor,
    lora: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
) -> torch.Tensor:
    """
    Reference SGMV: output[start:end] = input[start:end] @ lora[adapter_id].T

    Args:
        input: [M, K]
        lora: [num_adapters, N, K] - transposed layout
        lora_ids: [num_groups] adapter ID per group
        offsets: [num_groups + 1] token boundaries
    Returns:
        [M, N]
    """
    M, N = input.shape[0], lora.shape[1]
    output = torch.zeros(M, N, dtype=input.dtype, device=input.device)

    for g in range(len(lora_ids)):
        start, end = offsets[g], offsets[g + 1]
        adapter_id = lora_ids[g]
        if adapter_id >= 0 and end > start:
            output[start:end] = input[start:end] @ lora[adapter_id].T

    return output


def run_sgmv_kernel(
    input: torch.Tensor,
    lora: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
    max_lora_seq_len: int,
) -> torch.Tensor:
    session, device_ref, device = create_session()

    num_groups = len(lora_ids)

    with Graph(
        "sgmv_kernel_test",
        input_types=[
            TensorType(DTYPE, ["M", "K"], device=device_ref),
            TensorType(DTYPE, ["num_adapters", "N", "K"], device=device_ref),
            TensorType(DType.int32, ["lora_ids"], device=device_ref),
            TensorType(DType.uint32, ["lora_ranks"], device=DeviceRef.CPU()),
            TensorType(
                DType.uint32, ["lora_grouped_offsets"], device=device_ref
            ),
        ],
    ) as graph:
        x, lora_in, ids, ranks, offs = graph.inputs
        output = sgmv_kernel(
            input=x.tensor,
            lora=lora_in.tensor,
            lora_ids=ids.tensor,
            lora_ranks=ranks.tensor,
            input_row_offsets=offs.tensor,
            max_lora_seq_len=max_lora_seq_len,
        )
        graph.output(output)

    compiled = session.load(graph)
    result = compiled.execute(
        to_max_tensor(input, device),
        to_max_tensor(lora, device),
        Buffer.from_numpy(lora_ids.astype(np.int32)).to(device),
        Buffer.from_numpy(np.ones(num_groups, dtype=np.uint32)),
        Buffer.from_numpy(offsets.astype(np.uint32)).to(device),
    )
    return to_torch(result[0])


# -----------------------------------------------------------------------------
# Verification Functions
# -----------------------------------------------------------------------------


def verify_sgmv_kernel(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    K: int = 256,
    N: int = 64,
    seed: int = 42,
) -> None:
    """
    Verify sgmv_kernel correctness for a given configuration.

    Args:
        seq_lens: List of sequence lengths (all sequences, LoRA first then non-LoRA)
        lora_ids: List of adapter IDs for LoRA sequences only.
                  Length determines how many sequences have LoRA applied.
        num_adapters: Number of distinct LoRA adapters
        K: Input dimension
        N: Output dimension
        seed: Random seed

    Note: The kernel only processes LoRA sequences. Non-LoRA rows in the output
    buffer are not guaranteed to be zeros, so we only verify the LoRA portion.
    """
    num_lora_seqs = len(lora_ids)
    lora_seq_lens = seq_lens[:num_lora_seqs]

    lora_end = sum(lora_seq_lens)
    M = sum(seq_lens)

    input_tensor = rand_tensor((M, K), seed=seed)
    lora = rand_tensor((num_adapters, N, K), seed=seed + 1)

    lora_ids_arr = np.array(lora_ids, dtype=np.int32)
    offsets = calc_input_row_offsets(lora_seq_lens)

    actual = run_sgmv_kernel(
        input_tensor,
        lora,
        lora_ids_arr,
        offsets,
        max_lora_seq_len=max(lora_seq_lens) if lora_seq_lens else 1,
    )
    expected = torch_sgmv_kernel(input_tensor, lora, lora_ids_arr, offsets)

    if lora_end > 0:
        assert_close(actual[:lora_end], expected[:lora_end])
        assert actual[:lora_end].abs().sum() > 0, (
            "LoRA output should be non-zero"
        )


# -----------------------------------------------------------------------------
# SGMV Kernel Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens, lora_ids, num_adapters, K, N",
    [
        # Single sequence with single adapter
        ([16], [0], 1, 128, 64),
        # Multiple sequences, same adapter
        ([8, 12, 10], [0, 0, 0], 1, 128, 64),
        # Multiple sequences, different adapters
        ([80, 120, 100, 140], [0, 1, 2, 3], 4, 128, 256),
        # 3 LoRA + 3 non-LoRA, single adapter
        ([18, 10, 12, 8, 10, 12], [0, 0, 0], 1, 256, 64),
        # 4 LoRA + 2 non-LoRA, multiple adapters
        ([18, 10, 12, 14, 8, 10], [0, 1, 2, 1], 3, 256, 64),
    ],
)
def test_sgmv_kernel(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    K: int,
    N: int,
) -> None:
    """Test sgmv_kernel with various configurations."""
    verify_sgmv_kernel(
        seq_lens=seq_lens,
        lora_ids=lora_ids,
        num_adapters=num_adapters,
        K=K,
        N=N,
    )
