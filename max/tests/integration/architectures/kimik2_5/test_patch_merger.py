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
"""Tests for Kimi K2.5 patch merger."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.kimik2_5.layers.patch_merger import (
    PatchMergerMLP,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

MM_HIDDEN_SIZE = 1152
HIDDEN_SIZE = 1152
MERGE_KERNEL_SIZE = (2, 2)
EPS = 1e-05

N_K = MERGE_KERNEL_SIZE[0] * MERGE_KERNEL_SIZE[1]
INPUT_DIM = MM_HIDDEN_SIZE * N_K


def _generate_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


class TorchPatchMergerMLP(nn.Module):
    """PyTorch reference for PatchMergerMLP."""

    def __init__(self):
        super().__init__()
        self.hidden_size = INPUT_DIM
        self.pre_norm = nn.LayerNorm(MM_HIDDEN_SIZE, eps=EPS)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, HIDDEN_SIZE),
        )

    def forward(
        self, x: list[torch.Tensor], *args, **kwargs
    ) -> list[torch.Tensor]:
        x = [
            self.proj(self.pre_norm(item).view(item.shape[0], -1)) for item in x
        ]
        return x


def _create_weights() -> dict[str, torch.Tensor]:
    return {
        "pre_norm.weight": _generate_tensor((MM_HIDDEN_SIZE,)),
        "pre_norm.bias": _generate_tensor((MM_HIDDEN_SIZE,)),
        "proj.0.weight": _generate_tensor((INPUT_DIM, INPUT_DIM)),
        "proj.0.bias": _generate_tensor((INPUT_DIM,)),
        "proj.2.weight": _generate_tensor((HIDDEN_SIZE, INPUT_DIM)),
        "proj.2.bias": _generate_tensor((HIDDEN_SIZE,)),
    }


def _create_patch_merger_mlp(device: DeviceRef) -> PatchMergerMLP:
    return PatchMergerMLP(
        dtype=MAX_DTYPE,
        device=device,
        mm_hidden_size=MM_HIDDEN_SIZE,
        hidden_size=HIDDEN_SIZE,
        merge_kernel_size=MERGE_KERNEL_SIZE,
        eps=EPS,
    )


_PROJ_TO_MAX_KEYS = {
    "proj.0.weight": "linear1.weight",
    "proj.0.bias": "linear1.bias",
    "proj.2.weight": "linear2.weight",
    "proj.2.bias": "linear2.bias",
}


def _remap_keys_for_max(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {_PROJ_TO_MAX_KEYS.get(k, k): v for k, v in state_dict.items()}


def _build_and_run(
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
) -> Buffer:
    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)

    mlp = _create_patch_merger_mlp(device_ref)
    mlp.load_state_dict(_remap_keys_for_max(state_dict))

    session = InferenceSession(devices=[device])

    with Graph(
        "kimik2_5_patch_merger_mlp_test",
        input_types=[
            TensorType(MAX_DTYPE, tuple(x.shape), device=DeviceRef.GPU()),
        ],
    ) as graph:
        (graph_input,) = graph.inputs
        assert isinstance(graph_input, TensorValue)
        graph.output(mlp(graph_input))

    compiled = session.load(graph, weights_registry=mlp.state_dict())
    x_gpu = Buffer.from_dlpack(x).to(device)
    (result,) = compiled.execute(x_gpu)
    assert isinstance(result, Buffer)
    return result


def _assert_close(expected: torch.Tensor, actual: Buffer) -> None:
    rtol = 1e-2
    atol = 4 * torch.finfo(TORCH_DTYPE).eps
    torch.testing.assert_close(
        expected.cpu(),
        torch.from_dlpack(actual).cpu(),
        rtol=rtol,
        atol=atol,
    )


def test_patch_merger_mlp_gpu() -> None:
    """Test PatchMergerMLP accuracy on GPU against PyTorch reference."""
    torch.manual_seed(42)
    # Two ragged items with different patch counts: (N_i, N_k, mm_hidden_size)
    num_patches_a = 64
    num_patches_b = 24

    state_dict = _create_weights()
    item_a = _generate_tensor((num_patches_a, N_K, MM_HIDDEN_SIZE))
    item_b = _generate_tensor((num_patches_b, N_K, MM_HIDDEN_SIZE))

    # Concatenate into a single ragged tensor for MAX.
    x = torch.cat([item_a, item_b], dim=0)
    max_output = _build_and_run(state_dict, x)

    # Run torch reference per-item.
    ref = TorchPatchMergerMLP()
    ref.load_state_dict(state_dict)
    ref = ref.to(TORCH_DTYPE)
    torch_outputs = ref([item_a, item_b])
    torch_output = torch.cat(torch_outputs, dim=0).detach()

    _assert_close(torch_output, max_output)
