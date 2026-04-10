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
"""Tests for Gemma4 vision pooling."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    VISION_DEFAULT_OUTPUT_LENGTH,
    VISION_HIDDEN_SIZE,
    VISION_POOLING_KERNEL_SIZE,
    TorchGemma4VisionPooler,
)
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.vision_model.pooling import (
    Gemma4VisionPooler,
    avg_pool_by_positions,
)

TORCH_DTYPE = torch.bfloat16

_INPUT_SEQ_LEN = VISION_DEFAULT_OUTPUT_LENGTH * VISION_POOLING_KERNEL_SIZE**2
_GRID_W = 42
_GRID_H = 60


def _make_grid(w: int, h: int) -> np.ndarray:
    """Build a regular (x, y) patch-position grid, shape [w*h, 2]."""
    return np.array(
        [[x, y] for y in range(h) for x in range(w)],
        dtype=np.int32,
    )


def _buf_to_torch(buf: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buf).cpu().float()


# ---------------------------------------------------------------------------
# avg_pool_by_positions (NumPy) vs torch reference
# ---------------------------------------------------------------------------


class TestAvgPoolByPositions:
    """Tests for the NumPy avg_pool_by_positions function."""

    @pytest.mark.parametrize(
        "grid_w, grid_h, output_length, k",
        [
            (6, 6, 4, 3),
            (3, 3, 1, 3),
            (3, 6, 2, 3),
        ],
        ids=[
            "6x6-to-2x2",
            "3x3-to-1x1",
            "3x6-to-1x2",
        ],
    )
    def test_weights_match_torch_reference(
        self, grid_w: int, grid_h: int, output_length: int, k: int
    ) -> None:
        """NumPy weights must match the torch reference _avg_pool_by_positions."""
        hidden_size = 8
        input_seq_len = grid_w * grid_h

        positions_2d = _make_grid(grid_w, grid_h)
        assert positions_2d.shape == (input_seq_len, 2)

        np_weights = avg_pool_by_positions([positions_2d], [output_length], k)

        torch_ref = TorchGemma4VisionPooler(hidden_size, output_length)
        x_eye = torch.eye(input_seq_len, dtype=torch.float32).unsqueeze(0)
        positions_torch = torch.from_numpy(positions_2d).unsqueeze(0)
        ref_output, _ = torch_ref._avg_pool_by_positions(
            x_eye, positions_torch, output_length
        )
        ref_weights = ref_output[0].numpy()

        np.testing.assert_allclose(np_weights, ref_weights, atol=1e-6)

    def test_no_pooling_path(self) -> None:
        """When k=1, weights are identity (no spatial pooling)."""
        seq_len = 9
        hidden_size = 8

        side = int(seq_len**0.5)
        positions_np = _make_grid(side, side)

        weights = avg_pool_by_positions([positions_np], [seq_len], k=1)

        # k=1 means each patch maps 1:1 to an output bin.
        # Weight matrix should be a permutation with values 1/1**2=1.0.
        assert weights.shape == (seq_len, seq_len)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(weights.sum(axis=0), 1.0, atol=1e-6)

        # Verify our weights give the same hidden_states (scaled by
        # sqrt(hidden_size)) as the torch reference passthrough path.
        torch.manual_seed(99)
        x = torch.randn(1, seq_len, hidden_size, dtype=TORCH_DTYPE)

        # Compute np_pooled BEFORE the torch reference, because the torch
        # passthrough path uses ``hidden_states *= root_hidden_size`` which
        # mutates x in-place.
        np_pooled = (
            torch.from_numpy(weights).float()
            @ x[0].float()
            * math.sqrt(hidden_size)
        ).to(TORCH_DTYPE)

        torch_pooler = TorchGemma4VisionPooler(hidden_size, seq_len)
        padding = torch.ones(1, seq_len, dtype=torch.bool)
        positions_torch = torch.from_numpy(positions_np).unsqueeze(0)
        ref_output, _ref_mask = torch_pooler(
            x, positions_torch, padding, seq_len
        )

        torch.testing.assert_close(
            np_pooled, ref_output[0], rtol=2e-2, atol=2e-2
        )

    def test_non_square_grid(self) -> None:
        """Weights must be correct for a non-square rectangular grid."""
        # 6 wide * 4 tall = 24 patches, k=2, output = 24/4 = 6 bins.
        grid_w, grid_h = 6, 4
        input_seq_len = grid_w * grid_h
        output_length = 6
        k = 2

        positions_np = _make_grid(grid_w, grid_h)

        np_weights = avg_pool_by_positions([positions_np], [output_length], k)

        np.testing.assert_allclose(
            np_weights.sum(axis=1), np.ones(output_length), atol=1e-6
        )
        np.testing.assert_allclose(
            np_weights.sum(axis=0), np.full(input_seq_len, 0.25), atol=1e-6
        )

    def test_ragged_two_images(self) -> None:
        """Block-diagonal weights for two images of different sizes."""
        k = 2
        pos1 = _make_grid(4, 4)
        out1 = 4
        pos2 = _make_grid(6, 6)
        out2 = 9
        weights = avg_pool_by_positions([pos1, pos2], [out1, out2], k=k)

        total_patches = 16 + 36
        total_output = out1 + out2
        assert weights.shape == (total_output, total_patches)

        # Block-diagonal: image 1 block is top-left, image 2 block is
        # bottom-right. Cross-image entries must be zero.
        assert weights[:out1, 16:].sum() == 0.0, "no cross-image leakage"
        assert weights[out1:, :16].sum() == 0.0, "no cross-image leakage"

        # Each block row should sum to 1 (full average).
        np.testing.assert_allclose(
            weights.sum(axis=1), np.ones(total_output), atol=1e-6
        )

    def test_ragged_three_images_matches_individual(self) -> None:
        """Ragged batch of 3 images must equal per-image results assembled."""
        k = 3
        grids = [_make_grid(6, 6), _make_grid(3, 3), _make_grid(6, 3)]
        outputs = [4, 1, 2]

        # Combined.
        combined = avg_pool_by_positions(grids, outputs, k)

        # Per-image, then assemble block-diagonal.
        blocks = [
            avg_pool_by_positions([g], [o], k)
            for g, o in zip(grids, outputs, strict=False)
        ]
        patch_counts = [g.shape[0] for g in grids]
        total_patches = sum(patch_counts)
        total_output = sum(outputs)
        expected = np.zeros((total_output, total_patches), dtype=np.float32)

        row_off, col_off = 0, 0
        for block, n_out, n_patches in zip(
            blocks, outputs, patch_counts, strict=False
        ):
            expected[
                row_off : row_off + n_out, col_off : col_off + n_patches
            ] = block
            row_off += n_out
            col_off += n_patches

        np.testing.assert_allclose(combined, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Gemma4VisionPooler (MAX graph) vs torch reference forward
# ---------------------------------------------------------------------------


class TestGemma4VisionPooler:
    """Tests for the MAX Gemma4VisionPooler graph module."""

    def test_output_shape(self) -> None:
        """Pooler output must be [num_pooled, hidden_size]."""
        total_patches = 8
        num_pooled = 4
        hidden_size = 16
        pooler = Gemma4VisionPooler(hidden_size=hidden_size)

        device = CPU()
        dev_ref = DeviceRef.CPU()
        session = InferenceSession(devices=[device])

        with Graph(
            "pooler_shape",
            input_types=[
                TensorType(
                    DType.bfloat16,
                    [total_patches, hidden_size],
                    device=dev_ref,
                ),
                TensorType(
                    DType.bfloat16,
                    [num_pooled, total_patches],
                    device=dev_ref,
                ),
            ],
        ) as graph:
            h_in, w_in = graph.inputs
            assert isinstance(h_in, TensorValue)
            assert isinstance(w_in, TensorValue)
            graph.output(pooler(h_in, w_in))

        compiled = session.load(graph, weights_registry={})

        h = torch.randn(total_patches, hidden_size, dtype=TORCH_DTYPE)
        w = torch.rand(num_pooled, total_patches, dtype=TORCH_DTYPE)
        (result,) = compiled.execute(
            Buffer.from_dlpack(h), Buffer.from_dlpack(w)
        )

        assert _buf_to_torch(result).shape == (num_pooled, hidden_size)

    def test_values_match_torch_reference(self) -> None:
        """Full pooler output must match torch reference forward."""
        hidden_size = 16
        input_seq_len = 36
        output_length = 4
        k = 3
        batch_size = 1

        positions_np = _make_grid(6, 6)

        torch.manual_seed(42)
        x = torch.randn(
            batch_size, input_seq_len, hidden_size, dtype=TORCH_DTYPE
        )

        torch_pooler = TorchGemma4VisionPooler(hidden_size, output_length)
        positions_torch = torch.from_numpy(positions_np).unsqueeze(0)
        padding = torch.ones(batch_size, output_length, dtype=torch.bool)
        ref_output, _ref_mask = torch_pooler(
            x, positions_torch, padding, output_length
        )

        np_weights = avg_pool_by_positions([positions_np], [output_length], k)

        pooler = Gemma4VisionPooler(hidden_size=hidden_size)
        device = CPU()
        dev_ref = DeviceRef.CPU()
        session = InferenceSession(devices=[device])

        with Graph(
            "pooler_ref",
            input_types=[
                TensorType(
                    DType.bfloat16,
                    [input_seq_len, hidden_size],
                    device=dev_ref,
                ),
                TensorType(
                    DType.bfloat16,
                    [output_length, input_seq_len],
                    device=dev_ref,
                ),
            ],
        ) as graph:
            h_in, w_in = graph.inputs
            assert isinstance(h_in, TensorValue)
            assert isinstance(w_in, TensorValue)
            graph.output(pooler(h_in, w_in))

        compiled = session.load(graph, weights_registry={})

        x_unbatched = x[0]
        w_bf16 = torch.from_numpy(np_weights).to(TORCH_DTYPE)

        (result,) = compiled.execute(
            Buffer.from_dlpack(x_unbatched),
            Buffer.from_dlpack(w_bf16),
        )
        max_out = _buf_to_torch(result).to(TORCH_DTYPE)

        ref_unbatched = ref_output[0]

        torch.testing.assert_close(max_out, ref_unbatched, rtol=2e-2, atol=2e-2)

    def test_scale_factor(self) -> None:
        """Output must be scaled by sqrt(hidden_size)."""
        hidden_size = 16
        total_patches = 4
        num_pooled = 2

        pooler = Gemma4VisionPooler(hidden_size=hidden_size)

        device = CPU()
        dev_ref = DeviceRef.CPU()
        session = InferenceSession(devices=[device])

        with Graph(
            "pooler_scale",
            input_types=[
                TensorType(
                    DType.bfloat16,
                    [total_patches, hidden_size],
                    device=dev_ref,
                ),
                TensorType(
                    DType.bfloat16,
                    [num_pooled, total_patches],
                    device=dev_ref,
                ),
            ],
        ) as graph:
            h_in, w_in = graph.inputs
            assert isinstance(h_in, TensorValue)
            assert isinstance(w_in, TensorValue)
            graph.output(pooler(h_in, w_in))

        compiled = session.load(graph, weights_registry={})

        torch.manual_seed(1)
        h = torch.randn(total_patches, hidden_size, dtype=TORCH_DTYPE)
        # Identity-like weight: select first two rows.
        pw = torch.zeros(num_pooled, total_patches, dtype=TORCH_DTYPE)
        pw[0, 0] = 1.0
        pw[1, 1] = 1.0

        (result,) = compiled.execute(
            Buffer.from_dlpack(h), Buffer.from_dlpack(pw)
        )
        max_out = _buf_to_torch(result).to(TORCH_DTYPE)

        ref = (pw.float() @ h.float() * math.sqrt(hidden_size)).to(TORCH_DTYPE)
        torch.testing.assert_close(max_out, ref, rtol=2e-2, atol=2e-2)

    def test_ragged_two_images(self) -> None:
        """Pooler with ragged block-diagonal weights for two images."""
        hidden_size = 16
        k = 2

        pos1 = _make_grid(4, 4)
        n1, out1 = 16, 4
        pos2 = _make_grid(6, 6)
        n2, out2 = 36, 9

        total_patches = n1 + n2
        total_output = out1 + out2

        np_weights = avg_pool_by_positions([pos1, pos2], [out1, out2], k)

        torch.manual_seed(77)
        h = torch.randn(total_patches, hidden_size, dtype=TORCH_DTYPE)

        pooler = Gemma4VisionPooler(hidden_size=hidden_size)
        device = CPU()
        dev_ref = DeviceRef.CPU()
        session = InferenceSession(devices=[device])

        with Graph(
            "pooler_ragged",
            input_types=[
                TensorType(
                    DType.bfloat16,
                    [total_patches, hidden_size],
                    device=dev_ref,
                ),
                TensorType(
                    DType.bfloat16,
                    [total_output, total_patches],
                    device=dev_ref,
                ),
            ],
        ) as graph:
            h_in, w_in = graph.inputs
            assert isinstance(h_in, TensorValue)
            assert isinstance(w_in, TensorValue)
            graph.output(pooler(h_in, w_in))

        compiled = session.load(graph, weights_registry={})
        w_bf16 = torch.from_numpy(np_weights).to(TORCH_DTYPE)
        (result,) = compiled.execute(
            Buffer.from_dlpack(h), Buffer.from_dlpack(w_bf16)
        )
        max_out = _buf_to_torch(result).to(TORCH_DTYPE)

        torch_pooler1 = TorchGemma4VisionPooler(hidden_size, out1)
        torch_pooler2 = TorchGemma4VisionPooler(hidden_size, out2)
        h1 = h[:n1].unsqueeze(0).clone()
        h2 = h[n1:].unsqueeze(0).clone()
        ref1, _ = torch_pooler1(
            h1,
            torch.from_numpy(pos1).unsqueeze(0),
            torch.ones(1, out1, dtype=torch.bool),
            out1,
        )
        ref2, _ = torch_pooler2(
            h2,
            torch.from_numpy(pos2).unsqueeze(0),
            torch.ones(1, out2, dtype=torch.bool),
            out2,
        )
        ref_combined = torch.cat([ref1[0], ref2[0]], dim=0)

        torch.testing.assert_close(max_out, ref_combined, rtol=2e-2, atol=2e-2)

    def test_production_dimensions(self) -> None:
        """End-to-end test with gemma-4-31B-it production dimensions."""
        hidden_size = VISION_HIDDEN_SIZE
        input_seq_len = _INPUT_SEQ_LEN
        output_length = VISION_DEFAULT_OUTPUT_LENGTH
        k = VISION_POOLING_KERNEL_SIZE

        positions_np = _make_grid(_GRID_W, _GRID_H)
        assert positions_np.shape == (input_seq_len, 2)

        np_weights = avg_pool_by_positions([positions_np], [output_length], k)
        assert np_weights.shape == (output_length, input_seq_len)

        torch.manual_seed(7)
        x = torch.randn(1, input_seq_len, hidden_size, dtype=TORCH_DTYPE)
        torch_pooler = TorchGemma4VisionPooler(hidden_size, output_length)
        positions_torch = torch.from_numpy(positions_np).unsqueeze(0)
        padding = torch.ones(1, output_length, dtype=torch.bool)
        ref_output, _ref_mask = torch_pooler(
            x, positions_torch, padding, output_length
        )

        pooler = Gemma4VisionPooler(hidden_size=hidden_size)
        device = Accelerator(0)
        dev_ref = DeviceRef.GPU()
        session = InferenceSession(devices=[device])

        with Graph(
            "pooler_prod",
            input_types=[
                TensorType(
                    DType.bfloat16,
                    [input_seq_len, hidden_size],
                    device=dev_ref,
                ),
                TensorType(
                    DType.bfloat16,
                    [output_length, input_seq_len],
                    device=dev_ref,
                ),
            ],
        ) as graph:
            h_in, w_in = graph.inputs
            assert isinstance(h_in, TensorValue)
            assert isinstance(w_in, TensorValue)
            graph.output(pooler(h_in, w_in))

        compiled = session.load(graph, weights_registry={})

        x_unbatched = x[0]
        w_bf16 = torch.from_numpy(np_weights).to(TORCH_DTYPE)

        (result,) = compiled.execute(
            Buffer.from_dlpack(x_unbatched).to(device),
            Buffer.from_dlpack(w_bf16).to(device),
        )
        max_out = _buf_to_torch(result).to(TORCH_DTYPE)
        ref_unbatched = ref_output[0]

        torch.testing.assert_close(max_out, ref_unbatched, rtol=5e-2, atol=0.07)
