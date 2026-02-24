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
"""Tests for Kimi K2.5 rotary embeddings."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.pipelines.architectures.kimi2_5.layers.data_processing import (
    compute_position_ids,
)
from max.pipelines.architectures.kimi2_5.layers.rotary_embedding import (
    Rope2DPosEmbRepeated,
)

# ---------------------------------------------------------------------------
# Torch reference - direct translation of the Kimi-K2.5 reference code
# ---------------------------------------------------------------------------


class TorchRope2DPosEmbRepeated(nn.Module):
    """Nearly verbatim copy of the Kimi-K2.5 reference ``Rope2DPosEmbRepeated``."""

    def __init__(
        self,
        dim: int,
        max_height: int,
        max_width: int,
        theta_base: float = 10000,
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis(
        self, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        if not hasattr(self, "freqs_cis"):
            self.register_buffer(
                "freqs_cis",
                self._precompute_freqs_cis(device),
                persistent=False,
            )

        shapes = grid_thws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width
            for t, h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

DIM = 72
MAX_HEIGHT = 512
MAX_WIDTH = 512
THETA_BASE = 10000.0


def _build_and_run_precompute(
    dim: int,
    max_height: int,
    max_width: int,
    theta_base: float,
    device: Device,
) -> Buffer:
    """Build a graph that outputs the precomputed freqs_cis grid."""
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)

    rope = Rope2DPosEmbRepeated(
        dim=dim,
        max_height=max_height,
        max_width=max_width,
        theta_base=theta_base,
        device=device_ref,
    )

    with Graph("test_precompute", input_types=[]) as graph:
        result = rope.freqs_cis  # (N, dim//2, 2)
        graph.output(result)

    compiled = session.load(graph, weights_registry={})
    (buf,) = compiled.execute()
    return buf


def _build_and_run_get_freqs(
    dim: int,
    max_height: int,
    max_width: int,
    theta_base: float,
    position_ids: np.ndarray,
    device: Device,
) -> Buffer:
    """Build a graph that runs get_freqs_cis with the given position_ids."""
    session = InferenceSession(devices=[device])
    device_ref = DeviceRef.from_device(device)
    seq_len = len(position_ids)

    rope = Rope2DPosEmbRepeated(
        dim=dim,
        max_height=max_height,
        max_width=max_width,
        theta_base=theta_base,
        device=device_ref,
    )

    pos_ids_type = TensorType(DType.int64, shape=(seq_len,), device=device_ref)

    with Graph("test_get_freqs", input_types=[pos_ids_type]) as graph:
        (pos_ids_val,) = graph.inputs
        result = rope(pos_ids_val.tensor)
        graph.output(result)

    compiled = session.load(graph, weights_registry={})
    pos_ids_buf = Buffer.from_numpy(position_ids).to(device)
    (buf,) = compiled.execute(pos_ids_buf)
    return buf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPrecomputeFreqsCis:
    """Test that the precomputed frequency grid matches the torch reference."""

    def test_shape(self) -> None:
        buf = _build_and_run_precompute(
            DIM, MAX_HEIGHT, MAX_WIDTH, THETA_BASE, Accelerator()
        )
        result = torch.from_dlpack(buf)
        N = MAX_HEIGHT * MAX_WIDTH
        assert result.shape == (N, DIM // 2, 2)

    def test_values_match_torch(self) -> None:
        buf = _build_and_run_precompute(
            DIM, MAX_HEIGHT, MAX_WIDTH, THETA_BASE, Accelerator()
        )
        max_result = torch.from_dlpack(buf)  # (N, dim//2, 2)

        torch_ref = TorchRope2DPosEmbRepeated(
            DIM, MAX_HEIGHT, MAX_WIDTH, THETA_BASE
        )
        # Lazily initialise the buffer, then flatten to (N, dim//2) complex.
        grid_thws = torch.tensor([[1, MAX_HEIGHT, MAX_WIDTH]])
        ref_complex = torch_ref.get_freqs_cis(
            grid_thws, device=torch.device("cpu")
        )
        torch_result = torch.view_as_real(ref_complex)

        torch.testing.assert_close(
            max_result.cpu().float(),
            torch_result.cpu().float(),
            rtol=1e-4,
            atol=1e-4,
        )


class TestGetFreqsCis:
    """Test that gather-by-position-id matches the torch reference."""

    @pytest.mark.parametrize(
        "grid_thws",
        [
            [(1, 8, 8)],
            [(3, 16, 12)],
            [(1, 4, 4), (2, 8, 6)],
        ],
        ids=["single_square", "single_rect_repeated", "multi_video"],
    )
    def test_values_match_torch(
        self, grid_thws: list[tuple[int, int, int]]
    ) -> None:
        pos_ids = compute_position_ids(grid_thws, MAX_WIDTH)

        buf = _build_and_run_get_freqs(
            DIM, MAX_HEIGHT, MAX_WIDTH, THETA_BASE, pos_ids, Accelerator()
        )
        max_result = torch.from_dlpack(buf)  # (total_tokens, dim//2, 2)

        torch_ref = TorchRope2DPosEmbRepeated(
            DIM, MAX_HEIGHT, MAX_WIDTH, THETA_BASE
        )
        grid_thws_tensor = torch.tensor(grid_thws)
        torch_complex = torch_ref.get_freqs_cis(
            grid_thws_tensor, device=torch.device("cpu")
        )
        torch_result = torch.view_as_real(torch_complex)

        torch.testing.assert_close(
            max_result.cpu().float(),
            torch_result.cpu().float(),
            rtol=1e-5,
            atol=1e-5,
        )


class TestComputePositionIds:
    """Test the numpy position-id helper."""

    def test_single_video(self) -> None:
        ids = compute_position_ids([(1, 3, 4)], max_width=64)
        # Row 0: [0, 1, 2, 3], Row 1: [64, 65, 66, 67], Row 2: [128, 129, 130, 131]
        expected = np.array(
            [0, 1, 2, 3, 64, 65, 66, 67, 128, 129, 130, 131], dtype=np.int64
        )
        np.testing.assert_array_equal(ids, expected)

    def test_temporal_repeat(self) -> None:
        ids = compute_position_ids([(2, 2, 2)], max_width=64)
        single = np.array([0, 1, 64, 65], dtype=np.int64)
        expected = np.tile(single, 2)
        np.testing.assert_array_equal(ids, expected)

    def test_multi_video(self) -> None:
        ids = compute_position_ids([(1, 1, 2), (1, 2, 1)], max_width=64)
        expected = np.array([0, 1, 0, 64], dtype=np.int64)
        np.testing.assert_array_equal(ids, expected)
