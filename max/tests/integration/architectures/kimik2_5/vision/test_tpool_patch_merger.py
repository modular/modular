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
"""Tests verifying tpool_patch_merger graph kernel against the HuggingFace
Kimi-K2.5 reference implementation.

Reference: https://huggingface.co/nvidia/Kimi-K2.5-NVFP4/blob/main/modeling_kimi_k25.py
"""

from __future__ import annotations

import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import tpool_patch_merger

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16
RTOL = 2e-2
ATOL = 2 * torch.finfo(TORCH_DTYPE).eps

# Each entry is (grid_thws_data, merge_kernel_size, d_model) where
# grid_thws_data is a list of (T, H, W) per video.  H and W must be divisible
# by the corresponding kernel dims.
_CASES: list[tuple[list[list[int]], tuple[int, int], int]] = [
    # Single image (T=1), no temporal pooling.
    ([[1, 4, 4]], (2, 2), 64),
    # Single video (T>1): temporal mean-pooling reduces to one frame.
    ([[4, 8, 8]], (2, 2), 64),
    # Ragged batch: three videos with distinct T, H, W.
    ([[1, 4, 4], [2, 8, 8], [3, 4, 8]], (2, 2), 64),
    # Identity kernel: no spatial merging (kH=kW=1).
    ([[2, 4, 4]], (1, 1), 64),
    # Asymmetric kernel (kH=2, kW=4).
    ([[2, 8, 8]], (2, 4), 64),
    # Realistic single image from Kimi-K2.5 (98x148 patches, hidden_size=1152).
    ([[1, 98, 148]], (2, 2), 1152),
    # Realistic multi-image batch: three images at different resolutions.
    # Patch grids: 98x148 (1372x2072 px), 56x56 (784x784 px), 70x112 (980x1568 px).
    ([[1, 98, 148], [1, 56, 56], [1, 70, 112]], (2, 2), 1152),
]


def _reference_tpool_patch_merger(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """Reference matching HuggingFace ``tpool_patch_merger`` (modeling_kimi_k25.py).

    For each video ``(T, H, W)`` in ``grid_thws``:

    1. Slice ``T*H*W`` tokens from ``x``.
    2. View as ``(T, new_h, kH, new_w, kW, D)``.
    3. Permute to ``(T, new_h, new_w, kH, kW, D)`` and mean-pool over T.
    4. Flatten to ``(H * W, D)`` — equivalent to ``(new_h * new_w * kH * kW, D)``.

    Concatenates all per-video outputs along dim 0 to produce the flat 2-D
    tensor of shape ``[sum(H_i * W_i), D]`` that the MAX kernel returns.

    Args:
        x: Input tokens of shape ``(total_tokens, D)`` where
            ``total_tokens = sum(T_i * H_i * W_i)``.
        grid_thws: Per-video grid dimensions ``(T, H, W)``, shape ``(n, 3)``.
        merge_kernel_size: ``(kH, kW)`` spatial merge kernel.

    Returns:
        Flat tensor of shape ``[sum(H_i * W_i), D]``.
    """
    d_model = x.size(-1)
    kH, kW = merge_kernel_size
    outputs: list[torch.Tensor] = []
    pre_sum = 0
    for t, h, w in grid_thws.tolist():
        seq = x[pre_sum : pre_sum + t * h * w]
        new_h, new_w = h // kH, w // kW
        reshaped = seq.view(t, new_h, kH, new_w, kW, d_model)
        reshaped = reshaped.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(reshaped.reshape(new_h * new_w * kH * kW, d_model))
        pre_sum += t * h * w
    return torch.cat(outputs, dim=0)


def _build_and_run(
    x: torch.Tensor,
    grid_thws: torch.Tensor,
    merge_kernel_size: tuple[int, int],
) -> Buffer:
    """Compile and run a graph containing only ``tpool_patch_merger``.

    ``max_h`` and ``max_w`` are passed as integer constants (baked into the
    graph) so the test does not depend on ``ops.max`` graph construction.

    Args:
        x: Input tensor of shape ``(total_tokens, D)``, bfloat16.
        grid_thws: Grid dims ``(T, H, W)`` per video, shape ``(n, 3)``, int64.
        merge_kernel_size: ``(kH, kW)`` spatial merge kernel.

    Returns:
        Output :class:`Buffer` of shape ``[sum(H_i * W_i), D]``.
    """
    kH, kW = merge_kernel_size
    n_videos = grid_thws.shape[0]
    max_h = int(grid_thws[:, 1].max().item())
    max_w = int(grid_thws[:, 2].max().item())

    device = Accelerator(0)
    device_ref = DeviceRef.from_device(device)
    session = InferenceSession(devices=[device])

    with Graph(
        "tpool_patch_merger_test",
        input_types=[
            TensorType(MAX_DTYPE, tuple(x.shape), device=device_ref),
            TensorType(DType.int64, (n_videos, 3), device=device_ref),
        ],
    ) as graph:
        x_val, grid_val = graph.inputs
        assert isinstance(x_val, TensorValue)
        assert isinstance(grid_val, TensorValue)
        out = tpool_patch_merger(
            x_val,
            grid_val,
            kH=kH,
            kW=kW,
            max_h=max_h,
            max_w=max_w,
        )
        graph.output(out)

    compiled = session.load(graph)
    x_buf = Buffer.from_dlpack(x).to(device)
    grid_buf = Buffer.from_dlpack(grid_thws).to(device)
    (result,) = compiled.execute(x_buf, grid_buf)
    assert isinstance(result, Buffer)
    return result


@pytest.mark.parametrize(
    "grid_thws_data, merge_kernel_size, d_model",
    _CASES,
    ids=[
        "single_image",
        "single_video_multi_frame",
        "ragged_batch",
        "kernel_1x1",
        "kernel_2x4",
        "realistic_98x148_hidden1152",
        "realistic_multi_image_hidden1152",
    ],
)
def test_tpool_patch_merger(
    grid_thws_data: list[list[int]],
    merge_kernel_size: tuple[int, int],
    d_model: int,
) -> None:
    """MAX kernel output matches HuggingFace reference in shape and values.

    Checks two shapes:

    * The flat 2-D output ``[sum(H_i * W_i), D]`` returned directly by the
      kernel matches the analytically expected row count and the reference.
    * After the ``transformer.py`` reshape to ``[sum(H_i*W_i)//(kH*kW), kH*kW,
      D]``, the 3-D tensor matches the reference ``(new_h*new_w, kH*kW, D)``
      layout expected by ``PatchMergerMLP``.

    Args:
        grid_thws_data: Per-video ``(T, H, W)`` grid dimensions.
        merge_kernel_size: ``(kH, kW)`` spatial merge kernel.
    """
    torch.manual_seed(0)
    kH, kW = merge_kernel_size
    grid_thws = torch.tensor(grid_thws_data, dtype=torch.int64)
    total_tokens = int(sum(t * h * w for t, h, w in grid_thws_data))
    expected_rows = int(sum(h * w for _, h, w in grid_thws_data))

    x = torch.randn(total_tokens, d_model).to(TORCH_DTYPE)
    result = _build_and_run(x, grid_thws, merge_kernel_size)
    expected_flat = _reference_tpool_patch_merger(
        x, grid_thws, merge_kernel_size
    )

    actual_flat = torch.from_dlpack(result).cpu()

    # 2-D shape: [sum(H_i * W_i), D]
    assert actual_flat.shape == (expected_rows, d_model), (
        f"2-D shape mismatch: MAX={actual_flat.shape}, "
        f"expected=({expected_rows}, {d_model})"
    )
    assert actual_flat.shape == expected_flat.shape, (
        f"2-D shape mismatch vs reference: "
        f"MAX={actual_flat.shape}, reference={expected_flat.shape}"
    )

    torch.testing.assert_close(
        actual_flat, expected_flat.cpu(), rtol=RTOL, atol=ATOL
    )

    # 3-D reshape: [sum(H_i*W_i)//(kH*kW), kH*kW, D] = [sum(new_h_i*new_w_i), kH*kW, D]
    # This is the shape PatchMergerMLP receives after the transformer.py rebind+reshape.
    merge_k = kH * kW
    actual_3d = actual_flat.reshape(expected_rows // merge_k, merge_k, d_model)
    expected_3d = expected_flat.reshape(
        expected_rows // merge_k, merge_k, d_model
    )
    assert actual_3d.shape == expected_3d.shape, (
        f"3-D reshape mismatch: MAX={actual_3d.shape}, reference={expected_3d.shape}"
    )
    torch.testing.assert_close(
        actual_3d, expected_3d.cpu(), rtol=RTOL, atol=ATOL
    )
