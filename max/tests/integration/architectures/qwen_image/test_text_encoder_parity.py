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

"""Test that MAX Qwen2.5-VL text encoder matches HuggingFace Qwen2Model output.

Uses a tiny Qwen2 config (2 layers, small dims) with random weights to verify
the forward pass matches exactly. This catches:
- Missing final RMSNorm
- Wrong RoPE interleaving convention
- Weight loading mismatches
- Attention scale differences
"""

from __future__ import annotations

import numpy as np
import torch
from max.driver import Accelerator
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import DeviceRef, TensorType
from max.pipelines.architectures.qwen2_5vl.encoder.model_config import (
    Qwen25VLTextEncoderConfigBase,
)
from max.pipelines.architectures.qwen2_5vl.encoder.qwen25vl import (
    Qwen25VLTextEncoderTransformer,
)
from torch.utils.dlpack import from_dlpack
from transformers import Qwen2Config, Qwen2Model

# Small config for fast testing (2 layers, 256 hidden, 4 heads)
_HIDDEN_SIZE = 256
_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_NUM_LAYERS = 2
_INTERMEDIATE_SIZE = 512
_VOCAB_SIZE = 1024
_RMS_NORM_EPS = 1e-6
_ROPE_THETA = 1000000.0
_MAX_SEQ_LEN = 512
_HEAD_DIM = 64

_SEQ_LEN = 16
_SEED = 42


def _hf_config() -> Qwen2Config:
    return Qwen2Config(
        vocab_size=_VOCAB_SIZE,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        max_position_embeddings=_MAX_SEQ_LEN,
        rms_norm_eps=_RMS_NORM_EPS,
        rope_theta=_ROPE_THETA,
        use_sliding_window=False,
    )


def _max_config() -> Qwen25VLTextEncoderConfigBase:
    return Qwen25VLTextEncoderConfigBase(
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=_NUM_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        num_hidden_layers=_NUM_LAYERS,
        intermediate_size=_INTERMEDIATE_SIZE,
        vocab_size=_VOCAB_SIZE,
        rms_norm_eps=_RMS_NORM_EPS,
        rope_theta=_ROPE_THETA,
        max_seq_len=_MAX_SEQ_LEN,
        head_dim=_HEAD_DIM,
        device=DeviceRef.GPU(),
    )


def _build_hf_model() -> tuple[Qwen2Model, dict[str, torch.Tensor]]:
    """Create HF Qwen2Model with random weights and return model + state_dict."""
    torch.manual_seed(_SEED)
    model = Qwen2Model(_hf_config()).to(dtype=torch.bfloat16, device="cuda")
    model.eval()
    state_dict = dict(model.state_dict())
    return model, state_dict


def _build_max_model(
    hf_state_dict: dict[str, torch.Tensor],
) -> object:
    """Create MAX text encoder, compile with HF weights."""
    device_ref = Accelerator()

    with F.lazy():
        model = Qwen25VLTextEncoderTransformer(_max_config())
        model.to(device_ref)

    compiled = model.compile(
        TensorType(DType.int64, shape=[_SEQ_LEN], device=device_ref),
        weights=hf_state_dict,
    )
    return compiled


@torch.no_grad()
def _run_hf(model: Qwen2Model, token_ids: torch.Tensor) -> torch.Tensor:
    """Run HF model and return the last hidden state (after final norm)."""
    out = model(token_ids, output_hidden_states=True)
    # hidden_states[-1] is the output after the final RMSNorm
    return out.hidden_states[-1]


def _run_max(compiled_model: object, token_ids_np: np.ndarray) -> np.ndarray:
    """Run MAX model and return the last hidden state."""
    input_tensor = Tensor.from_dlpack(
        torch.tensor(token_ids_np, dtype=torch.int64, device="cuda")
    )
    result = compiled_model(input_tensor)  # type: ignore[operator]
    last_hs = result[-1]
    return np.from_dlpack(from_dlpack(last_hs).float().cpu())


def _random_tokens(seed_offset: int = 0) -> tuple[np.ndarray, torch.Tensor]:
    """Generate random token IDs as both numpy and torch tensors."""
    rng = np.random.RandomState(_SEED + seed_offset)
    token_ids_np = rng.randint(0, _VOCAB_SIZE, size=(_SEQ_LEN,)).astype(
        np.int64
    )
    token_ids_torch = torch.tensor(
        token_ids_np, dtype=torch.long, device="cuda"
    ).unsqueeze(0)
    return token_ids_np, token_ids_torch


def test_text_encoder_matches_hf() -> None:
    """Verify MAX text encoder output matches HF Qwen2Model for random weights."""
    hf_model, hf_state_dict = _build_hf_model()
    max_model = _build_max_model(hf_state_dict)

    token_ids_np, token_ids_torch = _random_tokens(seed_offset=1)

    hf_np = _run_hf(hf_model, token_ids_torch)[0].float().cpu().numpy()
    max_np = _run_max(max_model, token_ids_np)

    assert hf_np.shape == max_np.shape, (
        f"Shape mismatch: HF={hf_np.shape} vs MAX={max_np.shape}"
    )

    # Per-token cosine similarity
    for i in range(hf_np.shape[0]):
        cos = float(
            np.dot(hf_np[i], max_np[i])
            / (np.linalg.norm(hf_np[i]) * np.linalg.norm(max_np[i]) + 1e-10)
        )
        assert cos > 0.99, f"Token {i}: cosine similarity {cos:.6f} < 0.99"

    # Global cosine similarity
    cos_global = float(
        np.dot(hf_np.flatten(), max_np.flatten())
        / (np.linalg.norm(hf_np.flatten()) * np.linalg.norm(max_np.flatten()))
    )
    assert cos_global > 0.99, (
        f"Global cosine similarity {cos_global:.6f} < 0.99"
    )

    # Norm ratio should be close to 1.0
    hf_norms = np.linalg.norm(hf_np, axis=-1)
    max_norms = np.linalg.norm(max_np, axis=-1)
    norm_ratios = max_norms / (hf_norms + 1e-10)
    assert np.all(norm_ratios > 0.9) and np.all(norm_ratios < 1.1), (
        f"Norm ratio out of [0.9, 1.1]: "
        f"min={norm_ratios.min():.4f}, max={norm_ratios.max():.4f}"
    )


def test_text_encoder_norm_range() -> None:
    """Verify output norms match; catches a missing final RMSNorm (~2x off)."""
    hf_model, hf_state_dict = _build_hf_model()
    max_model = _build_max_model(hf_state_dict)

    token_ids_np, token_ids_torch = _random_tokens(seed_offset=2)

    hf_norms = (
        _run_hf(hf_model, token_ids_torch)[0].float().norm(dim=-1).cpu().numpy()
    )
    max_norms = np.linalg.norm(_run_max(max_model, token_ids_np), axis=-1)

    ratio = float(max_norms.mean()) / float(hf_norms.mean())
    assert 0.8 < ratio < 1.2, (
        f"Mean norm ratio {ratio:.4f} outside [0.8, 1.2]. "
        f"HF={float(hf_norms.mean()):.1f}, MAX={float(max_norms.mean()):.1f}. "
        f"Missing final RMSNorm?"
    )


def test_text_encoder_weight_count() -> None:
    """Verify all HF Qwen2Model parameters have a match in the MAX module."""
    with F.lazy():
        model = Qwen25VLTextEncoderTransformer(_max_config())

    max_param_names = set(name for name, _ in model.parameters)

    hf_model = Qwen2Model(_hf_config())
    hf_keys = set(hf_model.state_dict().keys())

    missing = hf_keys - max_param_names
    assert len(missing) == 0, (
        f"MAX model is missing {len(missing)} HF weight keys: "
        f"{sorted(missing)[:10]}"
    )
