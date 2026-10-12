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
    sgmv_lora_kernel,
    sgmv_lora_qkv_shrink,
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


def torch_sgmv_lora_kernel(
    input: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
) -> torch.Tensor:
    """
    Reference LoRA: output = input @ A.T @ B.T

    Args:
        input: [M, in_dim]
        lora_a: [num_adapters, rank, in_dim]
        lora_b: [num_adapters, out_dim, rank]
        lora_ids: [num_groups]
        offsets: [num_groups + 1]
    Returns:
        [M, out_dim]
    """
    v = torch_sgmv_kernel(input, lora_a, lora_ids, offsets)
    return torch_sgmv_kernel(v, lora_b, lora_ids, offsets)


def torch_sgmv_lora_qkv_shrink(
    input: torch.Tensor,
    lora_a: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
    lora_end_idx: int,
    max_rank: int,
) -> torch.Tensor:
    """
    Reference QKV shrink with planar output.

    Args:
        input: [M, K]
        lora_a: [num_adapters, 3*rank, K] - Q,K,V weights concatenated on dim 1
        lora_ids: [num_groups]
        offsets: [num_groups + 1]
        lora_end_idx: number of LoRA tokens to process
        max_rank: maximum rank for output padding
    Returns:
        [3, lora_end_idx, max_rank] - planar Q,K,V output
    """
    combined_rank = lora_a.shape[1]
    rank = combined_rank // 3
    input_lora = input[:lora_end_idx]

    flat_output = torch.zeros(
        lora_end_idx, combined_rank, dtype=input.dtype, device=input.device
    )

    for g in range(len(lora_ids)):
        start = offsets[g]
        end = min(offsets[g + 1], lora_end_idx)
        adapter_id = lora_ids[g]
        if adapter_id >= 0 and end > start:
            flat_output[start:end] = (
                input_lora[start:end] @ lora_a[adapter_id].T
            )

    # [M_lora, 3*rank] -> [M_lora, 3, rank] -> [3, M_lora, rank]
    output = (
        flat_output.reshape(lora_end_idx, 3, rank).permute(1, 0, 2).contiguous()
    )

    if rank < max_rank:
        padded = torch.zeros(
            3, lora_end_idx, max_rank, dtype=input.dtype, device=input.device
        )
        padded[:, :, :rank] = output
        output = padded

    return output


def run_sgmv_lora_kernel(
    input: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
    lora_end_idx: int,
    max_lora_seq_len: int,
) -> torch.Tensor:
    session, device_ref, device = create_session()

    _, rank, _ = lora_a.shape
    num_groups = len(lora_ids)

    with Graph(
        "sgmv_lora_kernel_test",
        input_types=[
            TensorType(DTYPE, ["M", "in_dim"], device=device_ref),
            TensorType(
                DTYPE, ["num_adapters", "rank", "in_dim"], device=device_ref
            ),
            TensorType(
                DTYPE, ["num_adapters", "out_dim", "rank"], device=device_ref
            ),
            TensorType(DType.int32, ["lora_ids"], device=device_ref),
            TensorType(DType.uint32, ["lora_ranks"], device=DeviceRef.CPU()),
            TensorType(
                DType.uint32, ["lora_grouped_offsets"], device=device_ref
            ),
            TensorType(DType.int64, ["lora_end"], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, a_in, b_in, ids, ranks, offs, end_idx = graph.inputs
        output = sgmv_lora_kernel(
            input=x.tensor,
            lora_a=a_in.tensor,
            lora_b=b_in.tensor,
            lora_ids=ids.tensor,
            lora_ranks=ranks.tensor,
            grouped_row_offsets=offs.tensor,
            lora_end_idx=end_idx.tensor,
            max_lora_seq_len=max_lora_seq_len,
        )
        graph.output(output)

    compiled = session.load(graph)
    lora_end_idx_arr = np.zeros(lora_end_idx, dtype=np.int64)
    if lora_end_idx > 0:
        lora_end_idx_arr[0] = lora_end_idx

    result = compiled.execute(
        to_max_tensor(input, device),
        to_max_tensor(lora_a, device),
        to_max_tensor(lora_b, device),
        Buffer.from_numpy(lora_ids.astype(np.int32)).to(device),
        Buffer.from_numpy(np.full(num_groups, rank, dtype=np.uint32)),
        Buffer.from_numpy(offsets.astype(np.uint32)).to(device),
        Buffer.from_numpy(lora_end_idx_arr),
    )
    return to_torch(result[0])


def run_sgmv_lora_qkv_shrink(
    input: torch.Tensor,
    lora_a: torch.Tensor,
    lora_ids: npt.NDArray[np.int32],
    offsets: npt.NDArray[np.uint32],
    lora_end_idx: int,
    max_lora_seq_len: int,
    max_rank: int,
) -> torch.Tensor:
    session, device_ref, device = create_session()

    with Graph(
        "sgmv_lora_qkv_shrink_test",
        input_types=[
            TensorType(DTYPE, ["M", "K"], device=device_ref),
            TensorType(
                DTYPE, ["num_adapters", "combined_rank", "K"], device=device_ref
            ),
            TensorType(DType.int32, ["lora_ids"], device=device_ref),
            TensorType(
                DType.uint32, ["lora_grouped_offsets"], device=device_ref
            ),
            TensorType(DType.int64, ["lora_end"], device=DeviceRef.CPU()),
        ],
    ) as graph:
        x, a_in, ids, offs, end_idx = graph.inputs
        output = sgmv_lora_qkv_shrink(
            input=x.tensor,
            lora_a=a_in.tensor,
            lora_ids=ids.tensor,
            lora_grouped_offsets=offs.tensor,
            lora_end_idx=end_idx.tensor,
            max_lora_seq_len=max_lora_seq_len,
            max_rank=max_rank,
        )
        graph.output(output)

    compiled = session.load(graph)
    lora_end_idx_arr = np.zeros(lora_end_idx, dtype=np.int64)
    if lora_end_idx > 0:
        lora_end_idx_arr[0] = lora_end_idx

    result = compiled.execute(
        to_max_tensor(input, device),
        to_max_tensor(lora_a, device),
        Buffer.from_numpy(lora_ids.astype(np.int32)).to(device),
        Buffer.from_numpy(offsets.astype(np.uint32)).to(device),
        Buffer.from_numpy(lora_end_idx_arr),
    )
    return to_torch(result[0])


# -----------------------------------------------------------------------------
# Verification Functions
# -----------------------------------------------------------------------------


def verify_sgmv_lora_kernel(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    in_dim: int = 256,
    out_dim: int = 256,
    rank: int = 8,
    seed: int = 42,
) -> None:
    """
    Verify sgmv_lora_kernel (shrink + expand) correctness.

    Args:
        seq_lens: List of sequence lengths (all sequences, LoRA first then non-LoRA)
        lora_ids: List of adapter IDs for LoRA sequences only.
        num_adapters: Number of distinct LoRA adapters
        in_dim: Input dimension
        out_dim: Output dimension
        rank: LoRA rank
        seed: Random seed

    Note: The kernel only processes LoRA sequences. Non-LoRA rows in the output
    buffer are not guaranteed to be zeros, so we only verify the LoRA portion.
    """
    num_lora_seqs = len(lora_ids)
    lora_seq_lens = seq_lens[:num_lora_seqs]

    lora_end = sum(lora_seq_lens)
    M = sum(seq_lens)

    input_tensor = rand_tensor((M, in_dim), seed=seed)
    lora_a = rand_tensor((num_adapters, rank, in_dim), seed=seed + 1)
    lora_b = rand_tensor((num_adapters, out_dim, rank), seed=seed + 2)

    lora_ids_arr = np.array(lora_ids, dtype=np.int32)
    offsets = calc_input_row_offsets(lora_seq_lens)

    actual = run_sgmv_lora_kernel(
        input_tensor,
        lora_a,
        lora_b,
        lora_ids_arr,
        offsets,
        lora_end_idx=lora_end,
        max_lora_seq_len=max(lora_seq_lens) if lora_seq_lens else 1,
    )
    expected = torch_sgmv_lora_kernel(
        input_tensor, lora_a, lora_b, lora_ids_arr, offsets
    )

    if lora_end > 0:
        assert_close(actual[:lora_end], expected[:lora_end])
        assert actual[:lora_end].abs().sum() > 0, (
            "LoRA output should be non-zero"
        )


def verify_qkv_shrink(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    K: int = 256,
    rank: int = 8,
    seed: int = 42,
) -> None:
    """
    Verify sgmv_lora_qkv_shrink correctness.

    Args:
        seq_lens: List of sequence lengths (all sequences, LoRA first then non-LoRA)
        lora_ids: List of adapter IDs for LoRA sequences only.
        num_adapters: Number of distinct LoRA adapters
        K: Input dimension
        rank: LoRA rank
        seed: Random seed
    """
    num_lora_seqs = len(lora_ids)
    lora_seq_lens = seq_lens[:num_lora_seqs]

    lora_end_idx = sum(lora_seq_lens)
    M = sum(seq_lens)

    input_tensor = rand_tensor((M, K), seed=seed)
    lora_a = rand_tensor((num_adapters, 3 * rank, K), seed=seed + 1)

    lora_ids_arr = np.array(lora_ids, dtype=np.int32)
    offsets = calc_input_row_offsets(lora_seq_lens)

    actual = run_sgmv_lora_qkv_shrink(
        input_tensor,
        lora_a,
        lora_ids_arr,
        offsets,
        lora_end_idx=lora_end_idx,
        max_lora_seq_len=max(lora_seq_lens) if lora_seq_lens else 1,
        max_rank=rank,
    )
    expected = torch_sgmv_lora_qkv_shrink(
        input_tensor,
        lora_a,
        lora_ids_arr,
        offsets,
        lora_end_idx=lora_end_idx,
        max_rank=rank,
    )

    assert actual.shape == (3, lora_end_idx, rank)
    assert_close(actual, expected)

    if lora_end_idx > 0:
        assert actual.abs().sum() > 0, "QKV shrink output should be non-zero"


# -----------------------------------------------------------------------------
# SGMV LoRA Kernel Tests (shrink + expand)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens, lora_ids, num_adapters, in_dim, out_dim, rank",
    [
        # Single sequence with single adapter
        ([16], [0], 1, 256, 256, 8),
        # Multiple sequences, same adapter
        ([8, 12, 10], [0, 0, 0], 1, 256, 256, 8),
        # Multiple sequences, different adapters
        ([8, 12, 10, 14], [0, 1, 2, 0], 3, 256, 256, 8),
        # 3 LoRA + 3 non-LoRA, single adapter
        ([8, 10, 12, 8, 10, 12], [0, 0, 0], 1, 256, 128, 16),
        # 4 LoRA + 2 non-LoRA, multiple adapters
        ([8, 23, 1, 14, 8, 10], [0, 1, 2, 1], 3, 128, 256, 16),
    ],
)
def test_sgmv_lora_kernel(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    in_dim: int,
    out_dim: int,
    rank: int,
) -> None:
    """Test sgmv_lora_kernel (shrink + expand) with various configurations."""
    verify_sgmv_lora_kernel(
        seq_lens=seq_lens,
        lora_ids=lora_ids,
        num_adapters=num_adapters,
        in_dim=in_dim,
        out_dim=out_dim,
        rank=rank,
    )


# -----------------------------------------------------------------------------
# QKV Shrink Tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_lens, lora_ids, num_adapters, K, rank",
    [
        # Single sequence with single adapter
        ([16], [0], 1, 256, 8),
        # Multiple sequences, same adapter
        ([8, 12, 10], [0, 0, 0], 1, 256, 8),
        # Multiple sequences, different adapters
        ([8, 107, 72, 14], [0, 1, 2, 0], 3, 256, 8),
        # 3 LoRA + 3 non-LoRA, single adapter
        ([8, 10, 12, 8, 10, 12], [0, 0, 0], 1, 512, 16),
        # 4 LoRA + 2 non-LoRA, multiple adapters
        ([8, 127, 35, 14, 800, 100], [0, 1, 1, 2], 3, 256, 16),
    ],
)
def test_qkv_shrink(
    seq_lens: list[int],
    lora_ids: list[int],
    num_adapters: int,
    K: int,
    rank: int,
) -> None:
    """Test sgmv_lora_qkv_shrink with various configurations."""
    verify_qkv_shrink(
        seq_lens=seq_lens,
        lora_ids=lora_ids,
        num_adapters=num_adapters,
        K=K,
        rank=rank,
    )
