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

"""Checks the Qwen3 text encoder's causal and additive paths.

The encoder's attention now derives the causal mask in the kernel and hands
the kernel grouped K/V heads (no ``_repeat_kv``). Both are invisible in the
output when they are right and quietly wrong when they are not: a mask that
is off by one, or Q heads mapped to the wrong K/V group, still produces
finite, plausible features. So this compares a small encoder against a
torch reference written from the module's own parameters, and separately
checks that the causal and additive-mask graphs agree with each other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from max import functional as F
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.qwen3_modulev3.text_encoder.model_config import (
    Qwen3TextEncoderConfig,
)
from max.pipelines.architectures.qwen3_modulev3.text_encoder.qwen3 import (
    Qwen3TextEncoderTransformer,
)
from max.pipelines.modeling.dataprocessing.causal_attention_mask import (
    causal_attention_mask_with_token_mask,
)
from max.tensor import Tensor

HIDDEN = 1024
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128
LAYERS = 2
INTERMEDIATE = 2048
VOCAB = 64
SEQ_LEN = 24
TAPS = (1, 2)


def _config() -> Qwen3TextEncoderConfig:
    return Qwen3TextEncoderConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=LAYERS,
        head_dim=HEAD_DIM,
        vocab_size=VOCAB,
        intermediate_size=INTERMEDIATE,
        rope_theta=10000.0,
        max_seq_len=64,
        rms_norm_eps=1e-6,
        dtype=DType.float32,
        device=DeviceRef.GPU(),
        hidden_state_layers=list(TAPS),
    )


def _build(
    device: Accelerator, weights_np: dict[str, np.ndarray], use_bias: bool
) -> Any:
    config = _config()
    with F.lazy():
        model = Qwen3TextEncoderTransformer(config, use_attention_bias=use_bias)
        model.to(device)
    # The encoder's modules hold bf16 parameters, so the weights go up in
    # the dtype the graph declares; numpy has no bfloat16, so torch carries
    # the cast.
    weights = {
        name: Tensor(
            storage=Buffer.from_dlpack(
                torch.from_numpy(weights_np[name]).to(torch.bfloat16)
            ).to(device)
        )
        for name, _ in model.parameters
    }
    assert set(weights) == set(weights_np), set(weights) ^ set(weights_np)
    return model.compile(*model.input_types(), weights=weights)


def _reference(
    weights: dict[str, np.ndarray],
    tokens: np.ndarray,
    *,
    mask_shift: int = 0,
) -> np.ndarray:
    """Torch reference of the encoder's forward, in float64.

    ``mask_shift`` moves the causal diagonal: 1 lets every row see one
    future key, so the intended output must sit much closer to
    ``mask_shift=0`` than to it.
    """
    # The reference sees exactly the bf16 values the GPU holds, widened.
    w = {
        k: torch.from_numpy(v).to(torch.bfloat16).double()
        for k, v in weights.items()
    }

    def rms_norm(x: torch.Tensor, name: str) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * w[name]

    def rope(x: torch.Tensor) -> torch.Tensor:
        half = HEAD_DIM // 2
        inv = 1.0 / (
            10000.0
            ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
        )
        freqs = torch.arange(SEQ_LEN, dtype=torch.float64)[:, None] * inv
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]

        def rotate_half(t: torch.Tensor) -> torch.Tensor:
            return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

        return x * cos + rotate_half(x) * sin

    h = w["embed_tokens.weight"][torch.from_numpy(tokens)]
    taps = []
    for layer in range(LAYERS):
        p = f"layers.{layer}."
        residual = h
        x = rms_norm(h, p + "input_layernorm.weight")
        q = (x @ w[p + "self_attn.q_proj.weight"].T).reshape(
            SEQ_LEN, HEADS, HEAD_DIM
        )
        k = (x @ w[p + "self_attn.k_proj.weight"].T).reshape(
            SEQ_LEN, KV_HEADS, HEAD_DIM
        )
        v = (x @ w[p + "self_attn.v_proj.weight"].T).reshape(
            SEQ_LEN, KV_HEADS, HEAD_DIM
        )
        q = rms_norm(q, p + "self_attn.q_norm.weight")
        k = rms_norm(k, p + "self_attn.k_norm.weight")
        q, k = rope(q), rope(k)
        n_rep = HEADS // KV_HEADS
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.einsum("shd,thd->hst", q, k) / HEAD_DIM**0.5
        causal = torch.tril(
            torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), diagonal=mask_shift
        )
        scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.einsum("hst,thd->shd", probs, v).reshape(
            SEQ_LEN, HEADS * HEAD_DIM
        )
        h = residual + attn @ w[p + "self_attn.o_proj.weight"].T

        residual = h
        x = rms_norm(h, p + "post_attention_layernorm.weight")
        gate = torch.nn.functional.silu(x @ w[p + "mlp.gate_proj.weight"].T)
        up = x @ w[p + "mlp.up_proj.weight"].T
        h = residual + (gate * up) @ w[p + "mlp.down_proj.weight"].T
        # The transformer's tap l is hidden_states[l] = output of block l-1.
        if layer + 1 in TAPS:
            taps.append(h)

    # The encoder emits hidden-major: each hidden unit's taps adjacent.
    return torch.stack(taps, dim=-1).flatten(-2)[None, ...].numpy()


def _to_numpy(t: Tensor) -> np.ndarray:
    """bf16 has no numpy dtype, so go through torch."""
    storage = t.storage
    assert storage is not None
    return torch.from_dlpack(storage).float().cpu().numpy()


def test_causal_matches_reference_and_additive() -> None:
    rng = np.random.default_rng(7)
    device = Accelerator()

    # The encoder's modules hold bf16 parameters, so the weights are bf16 and
    # the reference runs in float64 on exactly those values. A causal diagonal
    # off by one, or Q heads mapped to the wrong K/V group (repeat_interleave
    # above, not a tile), moves the output by O(1) relative -- far outside the
    # tolerance this comparison keeps for bf16 rounding.
    config = _config()
    with F.lazy():
        template = Qwen3TextEncoderTransformer(config)
    weights_np = {}
    for name, param in template.parameters:
        shape = tuple(int(d) for d in param.shape)
        weights_np[name] = (rng.standard_normal(shape) * 0.02).astype(
            np.float32
        )

    tokens = rng.integers(0, VOCAB, size=(SEQ_LEN,), dtype=np.int64)
    tokens_tensor = Tensor(tokens, device=device)

    causal_model = _build(device, weights_np, use_bias=False)
    out_causal = _to_numpy(causal_model(tokens_tensor)[0])

    bias_np = causal_attention_mask_with_token_mask(
        [0], np.ones((SEQ_LEN,), dtype=np.bool_)
    )
    bias = Tensor(
        storage=Buffer.from_numpy(bias_np[:, np.newaxis, :, :]).to(device)
    )
    additive_model = _build(device, weights_np, use_bias=True)
    out_additive = _to_numpy(additive_model(tokens_tensor, bias)[0])

    reference = _reference(weights_np, tokens)

    shifted = _reference(weights_np, tokens, mask_shift=1)

    assert out_causal.shape == (1, SEQ_LEN, len(TAPS) * HIDDEN)
    np.testing.assert_allclose(out_causal, out_additive, rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(out_causal, reference, rtol=3e-2, atol=3e-2)

    # The comparison is not vacuous: the same encoder against a causal mask
    # one key late is far off.
    wrong = np.abs(out_causal - shifted).max()
    right = np.abs(out_causal - reference).max()
    assert wrong > 10 * right, (wrong, right)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
