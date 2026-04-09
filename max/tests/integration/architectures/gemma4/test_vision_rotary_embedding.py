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
"""Unit tests for Gemma4 vision rotary position embeddings.

Each test exercises ``compute_vision_freqs_cis`` in isolation on CPU and
validates shape and numerical correctness against analytic expectations.
"""

from __future__ import annotations

import math

import torch
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    compute_vision_freqs_cis,
)


def _buf_to_torch(buf: Buffer) -> torch.Tensor:
    return torch.from_dlpack(buf).cpu().float()


def _hf_vision_rope_reference(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for Gemma4 multi-dimensional vision RoPE.

    Transcribed from ``Gemma4RotaryEmbedding.forward()`` in
    ``transformers/models/gemma4/modeling_gemma4.py`` (the ``ndim > 1``
    branch, lines 1570-1603). ``attention_scaling`` is omitted because it
    is always 1.0 for proportional RoPE.

    Args:
        position_ids: Integer grid coordinates, shape
            ``[batch, num_patches, ndim]``.
        head_dim: Full per-head dimension.
        theta: RoPE base frequency (``rope_theta``).

    Returns:
        ``(cos, sin)`` each of shape ``[batch, num_patches, head_dim]``.
        Within ``head_dim``, each spatial dimension contributes
        ``head_dim // ndim`` values arranged as
        ``[freqs, freqs_dup]`` (the HF "half-rotate" convention).
    """
    ndim = position_ids.shape[-1]
    head_dim_per_dim = head_dim // ndim
    all_embs: list[torch.Tensor] = []

    for d in range(ndim):
        dim_inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim_per_dim, 2, dtype=torch.float)
                / head_dim_per_dim
            )
        )
        dim_inv_freq_expanded = (
            dim_inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
        )
        dim_positions = position_ids[:, :, d]  # [batch, num_patches]
        dim_positions_expanded = dim_positions[:, None, :].float()

        dim_freqs = (
            dim_inv_freq_expanded.float() @ dim_positions_expanded.float()
        ).transpose(1, 2)
        dim_emb = torch.cat(
            (dim_freqs, dim_freqs), dim=-1
        )  # [batch, seq, 2*freq_per_dim]
        all_embs.append(dim_emb)

    emb = torch.cat(all_embs, dim=-1)
    cos = emb.cos()
    sin = emb.sin()
    return cos, sin


def test_compute_vision_freqs_cis_shape() -> None:
    """output shape must be [total_patches, head_dim // 2, 2]."""
    total_patches = 6
    head_dim = 8
    ndim = 2
    theta = 10000.0

    pos_ids = torch.tensor(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]], dtype=torch.int32
    )

    device = CPU()
    dev_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    with Graph(
        "freqs_cis_shape",
        input_types=[
            TensorType(DType.int32, [total_patches, 2], device=dev_ref)
        ],
    ) as graph:
        (inp,) = graph.inputs
        assert isinstance(inp, TensorValue)
        graph.output(
            compute_vision_freqs_cis(
                inp,
                head_dim=head_dim,
                ndim=ndim,
                theta=theta,
                dtype=DType.float32,
                device=dev_ref,
            )
        )

    compiled = session.load(graph, weights_registry={})
    (result,) = compiled.execute(pos_ids.numpy())

    assert _buf_to_torch(result).shape == (total_patches, head_dim // 2, 2)


def test_compute_vision_freqs_cis_values() -> None:
    """Cos/sin values must match analytic computation for a single patch."""
    # head_dim=4, ndim=2 → channels_per_dim=2, freqs_per_dim=1
    head_dim = 4
    ndim = 2
    theta = 10000.0
    channels_per_dim = 2 * (head_dim // (2 * ndim))  # 2
    inv_freq = 1.0 / (theta ** (0.0 / channels_per_dim))  # only one freq

    pos_ids = torch.tensor([[2, 3]], dtype=torch.int32)  # x=2, y=3

    device = CPU()
    dev_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    with Graph(
        "freqs_cis_values",
        input_types=[TensorType(DType.int32, [1, 2], device=dev_ref)],
    ) as graph:
        (inp,) = graph.inputs
        assert isinstance(inp, TensorValue)
        graph.output(
            compute_vision_freqs_cis(
                inp,
                head_dim=head_dim,
                ndim=ndim,
                theta=theta,
                dtype=DType.float32,
                device=dev_ref,
            )
        )

    compiled = session.load(graph, weights_registry={})
    (result,) = compiled.execute(pos_ids.numpy())
    result_t = _buf_to_torch(result)  # [1, 2, 2]

    expected = torch.tensor(
        [
            [math.cos(2 * inv_freq), math.sin(2 * inv_freq)],  # x-dim
            [math.cos(3 * inv_freq), math.sin(3 * inv_freq)],  # y-dim
        ],
        dtype=torch.float32,
    ).unsqueeze(0)  # [1, 2, 2]

    torch.testing.assert_close(result_t, expected, rtol=1e-5, atol=1e-5)


def test_compute_vision_freqs_cis_zero_position() -> None:
    """Position (0, 0) must give cos=1, sin=0 for all frequencies."""
    head_dim = 8
    ndim = 2
    theta = 10000.0

    pos_ids = torch.zeros(1, 2, dtype=torch.int32)

    device = CPU()
    dev_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    with Graph(
        "freqs_cis_zero",
        input_types=[TensorType(DType.int32, [1, 2], device=dev_ref)],
    ) as graph:
        (inp,) = graph.inputs
        assert isinstance(inp, TensorValue)
        graph.output(
            compute_vision_freqs_cis(
                inp,
                head_dim=head_dim,
                ndim=ndim,
                theta=theta,
                dtype=DType.float32,
                device=dev_ref,
            )
        )

    compiled = session.load(graph, weights_registry={})
    (result,) = compiled.execute(pos_ids.numpy())
    result_t = _buf_to_torch(result)  # [1, head_dim//2, 2]

    # cos(0)=1, sin(0)=0 for every frequency.
    assert (result_t[..., 0] - 1.0).abs().max() < 1e-6, "cos must be 1 at pos 0"
    assert result_t[..., 1].abs().max() < 1e-6, "sin must be 0 at pos 0"


def test_compute_vision_freqs_cis_matches_hf_reference() -> None:
    """MAX output must match the HuggingFace Gemma4 reference implementation."""
    head_dim = 16
    ndim = 2
    theta = 10000.0
    fpd = head_dim // (2 * ndim)  # freqs_per_dim

    pos_ids = torch.tensor(
        [[0, 0], [3, 1], [5, 7], [2, 4], [1, 6], [0, 3]],
        dtype=torch.int32,
    )
    total_patches = pos_ids.shape[0]

    # --- Run MAX graph ---
    device = CPU()
    dev_ref = DeviceRef.CPU()
    session = InferenceSession(devices=[device])

    with Graph(
        "freqs_cis_hf_ref",
        input_types=[
            TensorType(DType.int32, [total_patches, ndim], device=dev_ref)
        ],
    ) as graph:
        (inp,) = graph.inputs
        assert isinstance(inp, TensorValue)
        graph.output(
            compute_vision_freqs_cis(
                inp,
                head_dim=head_dim,
                ndim=ndim,
                theta=theta,
                dtype=DType.float32,
                device=dev_ref,
            )
        )

    compiled = session.load(graph, weights_registry={})
    (result,) = compiled.execute(pos_ids.numpy())
    max_out = _buf_to_torch(result)  # [total_patches, head_dim//2, 2]
    max_cos = max_out[..., 0]  # [total_patches, head_dim//2]
    max_sin = max_out[..., 1]

    # --- Run HF reference ---
    # HF expects [batch, num_patches, ndim]; use batch=1.
    hf_cos, hf_sin = _hf_vision_rope_reference(
        pos_ids.unsqueeze(0), head_dim, theta
    )
    # hf_cos/sin shape: [1, total_patches, head_dim]
    # Layout within head_dim: [dim0_freqs(fpd) | dim0_dup(fpd) | dim1_freqs(fpd) | dim1_dup(fpd)]
    # Extract the non-duplicated halves to match MAX's [total_patches, head_dim//2] layout.
    dim_size = 2 * fpd  # = head_dim // ndim
    hf_cos_unique = hf_cos[0].reshape(total_patches, ndim, dim_size)[:, :, :fpd]
    hf_cos_unique = hf_cos_unique.reshape(total_patches, ndim * fpd)
    hf_sin_unique = hf_sin[0].reshape(total_patches, ndim, dim_size)[:, :, :fpd]
    hf_sin_unique = hf_sin_unique.reshape(total_patches, ndim * fpd)

    torch.testing.assert_close(max_cos, hf_cos_unique, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(max_sin, hf_sin_unique, rtol=1e-5, atol=1e-5)
