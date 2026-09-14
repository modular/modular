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
"""Tests for the ModuleV3 Gemma4VisionAttention against the HF reference.

The reference is implemented in ``conftest.TorchGemma4VisionAttention`` to
avoid a hard dependency on the ``transformers.models.gemma4`` module (which
may not be present in all test environments).  The implementation is a
faithful port of the HF ``Gemma4VisionAttention.forward`` for the vision path:

* Linear Q/K/V projections (no bias)
* Q/K RMS-norm with learned weight (weight_offset=1)
* V RMS-norm with no learned weight (bare normalisation)
* 2-D multidimensional RoPE via the rotate-half convention
* Bidirectional scaled dot-product attention (scale=1.0, no causal mask)
* Output projection

The MAX implementation uses ``flash_attention_ragged_gpu`` on packed
(ragged) sequences; the reference uses eager attention.  Results must match
within the tolerances accepted for bfloat16 flash vs. eager attention.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from conftest import (
    VISION_EMBED_HIDDEN_SIZE,
    VISION_NUM_HEADS,
    VISION_RMS_NORM_EPS,
    TorchGemma4VisionAttention,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.gemma4_modulev3.vision_model.attention import (
    Gemma4VisionAttention,
)

# Head dim consistent with the small test dimensions.
VISION_HEAD_DIM = VISION_EMBED_HIDDEN_SIZE // VISION_NUM_HEADS

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

ATOL = 8 * torch.finfo(TORCH_DTYPE).eps
RTOL = 2e-2


def _make_max_vision_config() -> Any:
    """Minimal stub accepted by the ModuleV3 Gemma4VisionAttention.__init__."""
    vision_cfg = SimpleNamespace(
        hidden_size=VISION_EMBED_HIDDEN_SIZE,
        num_attention_heads=VISION_NUM_HEADS,
        num_key_value_heads=VISION_NUM_HEADS,
        head_dim=VISION_HEAD_DIM,
        rms_norm_eps=VISION_RMS_NORM_EPS,
        attention_bias=False,
    )
    return SimpleNamespace(
        devices=[DeviceRef.GPU()],
        unquantized_dtype=DType.bfloat16,
        vision_config=vision_cfg,
    )


@torch.no_grad()
def _run_ref_attention(
    ref: TorchGemma4VisionAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Run the reference attention on GPU and return output on CPU."""
    ref = ref.cuda().to(TORCH_DTYPE).eval()
    cos, sin = position_embeddings
    return ref(hidden_states.cuda(), (cos.cuda(), sin.cuda())).cpu()


def _build_and_run_max_attention(
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """Compile the ModuleV3 Gemma4VisionAttention and execute it."""
    num_patches = hidden_states.shape[0]
    device = Accelerator(0)
    with F.lazy(), default_dtype(MAX_DTYPE):
        attn = Gemma4VisionAttention(_make_max_vision_config(), layer_idx=0)
        attn.to(device)

    input_types = (
        TensorType(
            MAX_DTYPE,
            ("num_patches", VISION_EMBED_HIDDEN_SIZE),
            device=DeviceRef.GPU(),
        ),
        TensorType(
            MAX_DTYPE,
            ("num_patches", VISION_HEAD_DIM // 2, 2),
            device=DeviceRef.GPU(),
        ),
        TensorType(DType.uint32, ("seqlens_size",), device=DeviceRef.GPU()),
        TensorType(DType.uint32, shape=[], device=DeviceRef.CPU()),
    )
    compiled = attn.compile(*input_types, weights=weights)

    cu_seqlens = torch.tensor([0, num_patches], dtype=torch.uint32)
    max_seq_len = np.array(num_patches, dtype=np.uint32)
    result = compiled.execute_raw(
        Buffer.from_dlpack(hidden_states).to(device),
        Buffer.from_dlpack(freqs_cis).to(device),
        Buffer.from_dlpack(cu_seqlens).to(device),
        Buffer.from_numpy(max_seq_len),
    )[0]
    assert isinstance(result, Buffer)
    return torch.from_dlpack(result).cpu()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _random_weight(shape: tuple[int, ...]) -> torch.Tensor:
    return (torch.randn(shape) * (1.0 / math.sqrt(shape[-1]))).to(TORCH_DTYPE)


def _make_weights() -> dict[str, torch.Tensor]:
    """Random attention weights shared between MAX and reference."""
    nh, hd, hs = VISION_NUM_HEADS, VISION_HEAD_DIM, VISION_EMBED_HIDDEN_SIZE
    proj_out = nh * hd
    return {
        "q_proj.weight": _random_weight((proj_out, hs)),
        "k_proj.weight": _random_weight((proj_out, hs)),
        "v_proj.weight": _random_weight((proj_out, hs)),
        "o_proj.weight": _random_weight((hs, proj_out)),
        "q_norm.weight": _random_weight((hd,)),
        "k_norm.weight": _random_weight((hd,)),
    }


def _make_inputs(
    num_patches: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(hidden_states, freqs_cis, position_embeddings)`` test tensors."""
    hs = torch.randn(num_patches, VISION_EMBED_HIDDEN_SIZE, dtype=TORCH_DTYPE)
    H = VISION_HEAD_DIM
    freqs_per_dim = H // 4  # each spatial dim gets H//4 independent frequencies

    # Independent random angles for each spatial dimension.
    angles_d0 = torch.rand(num_patches, freqs_per_dim) * 2 * math.pi
    angles_d1 = torch.rand(num_patches, freqs_per_dim) * 2 * math.pi

    cos_d0 = torch.cos(angles_d0).to(TORCH_DTYPE)  # [p, H//4]
    sin_d0 = torch.sin(angles_d0).to(TORCH_DTYPE)
    cos_d1 = torch.cos(angles_d1).to(TORCH_DTYPE)
    sin_d1 = torch.sin(angles_d1).to(TORCH_DTYPE)

    # MAX format: [p, H//2, 2]  (first H//4 = dim0, next H//4 = dim1)
    freqs_cis = torch.stack(
        [
            torch.cat([cos_d0, cos_d1], dim=-1),
            torch.cat([sin_d0, sin_d1], dim=-1),
        ],
        dim=-1,
    )  # [p, H//2, 2]

    # HF format: each [p, H] = [dim0_cos, dim0_cos, dim1_cos, dim1_cos]
    # (frequencies repeated for the rotate-half / non-interleaved convention).
    cos_hf = torch.cat([cos_d0, cos_d0, cos_d1, cos_d1], dim=-1)
    sin_hf = torch.cat([sin_d0, sin_d0, sin_d1, sin_d1], dim=-1)

    return hs, freqs_cis, (cos_hf, sin_hf)


@torch.no_grad()
def test_vision_attention_matches_reference() -> None:
    """Verify Gemma4VisionAttention output matches the HF-ported reference.

    Both implementations share the same random weights and inputs.
    The MAX layer uses ``flash_attention_ragged_gpu``; the reference uses
    eager attention.  Both must agree within bfloat16 flash-attention
    tolerances.
    """
    torch.manual_seed(42)
    weights = _make_weights()
    hidden_states, freqs_cis, position_embeddings = _make_inputs(num_patches=16)

    ref = TorchGemma4VisionAttention(
        VISION_EMBED_HIDDEN_SIZE,
        VISION_NUM_HEADS,
        VISION_HEAD_DIM,
        VISION_RMS_NORM_EPS,
    )
    ref.load_state_dict(weights)
    ref_out = _run_ref_attention(ref, hidden_states, position_embeddings)

    max_out = _build_and_run_max_attention(weights, hidden_states, freqs_cis)

    torch.testing.assert_close(ref_out, max_out, rtol=RTOL, atol=ATOL)
