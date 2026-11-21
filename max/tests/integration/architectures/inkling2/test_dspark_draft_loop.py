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
"""Checks the Inkling 2 DSpark drafter's sequential draft loop on the host.

The reference transcribes vLLM's ``_sample_sequential`` with the Markov head's
``hidden_out`` folded into the LM head (``vllm/models/inkling2/speculator.py``
and ``dspark.py``) in NumPy, rounding to bfloat16 where they do. It runs apart
from ``test_dspark_numerics`` because each host graph compiles on first use.
"""

from __future__ import annotations

import numpy as np
from max import functional as F
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.inkling2.dspark import (
    Inkling2DSpark,
    Inkling2MarkovHead,
    construct_dspark_kv_params,
)
from max.pipelines.architectures.inkling2.dspark_config import (
    Inkling2DSparkConfig,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor import Tensor, default_device, default_dtype

HIDDEN, RANK, VOCAB, TAPS = 16, 8, 32, 3

CONFIG = Inkling2DSparkConfig(
    hidden_size=HIDDEN,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    intermediate_size=32,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    local_layer_ids=(),
    sliding_window_size=16,
    block_size=4,
    markov_rank=RANK,
    target_layer_ids=tuple(range(TAPS)),
    vocab_size=VOCAB,
)


def _bf16(x: np.ndarray) -> np.ndarray:
    """Rounds float32 to the nearest bfloat16, ties to even."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & (
        np.uint32(0xFFFF0000)
    )
    return rounded.astype(np.uint32).view(np.float32)


def _tensor(x: np.ndarray, dtype: DType) -> Tensor:
    return Tensor.from_dlpack(np.ascontiguousarray(x)).cast(dtype)


def _head(
    rng: np.random.Generator,
) -> tuple[Inkling2MarkovHead, dict[str, np.ndarray]]:
    weights = {
        "markov_w1.weight": rng.normal(size=(VOCAB, RANK)),
        "rnn_proj.weight": rng.normal(size=(3 * RANK, 2 * RANK + 2 * HIDDEN))
        * 0.2,
        "rnn_proj.bias": rng.normal(size=(3 * RANK,)) * 0.2,
        "hidden_out.weight": rng.normal(size=(HIDDEN, RANK)),
        "hidden_out.bias": rng.normal(size=(HIDDEN,)),
        "dflare_norm.weight": rng.normal(size=(HIDDEN,)),
    }
    weights = {k: _bf16(v.astype(np.float32)) for k, v in weights.items()}
    with default_dtype(DType.bfloat16), default_device(CPU()):
        head = Inkling2MarkovHead(CONFIG)
        head.load_state_dict(weights, auto_cast=True)
    return head, weights


def test_the_draft_loop_matches_the_reference_sampler() -> None:
    rng = np.random.default_rng(2)
    seqs, steps, unpadded, cap = 3, CONFIG.block_size - 1, VOCAB - 2, 30.0
    block_hidden = _bf16(
        rng.normal(size=(seqs * CONFIG.block_size, HIDDEN)).astype(np.float32)
    )
    f_rnn = _bf16(rng.normal(size=(seqs, HIDDEN)).astype(np.float32))
    lm_head = _bf16(rng.normal(size=(VOCAB, HIDDEN)).astype(np.float32))
    anchors = rng.integers(0, unpadded, size=(seqs,))

    kv_params = construct_dspark_kv_params(
        CONFIG, KVCacheConfig(), [DeviceRef.CPU()], DType.bfloat16
    )
    _, w = _head(rng)
    with default_dtype(DType.bfloat16), default_device(CPU()):
        with F.lazy():
            drafter = Inkling2DSpark(
                CONFIG, kv_params, max_seq_len=64, device=CPU()
            )
        weights = {
            name: np.zeros([int(d) for d in param.shape], np.float32)
            for name, param in drafter.parameters
        }
        weights.update({f"markov_head.{k}": v for k, v in w.items()})
        drafter.load_state_dict(weights, auto_cast=True)
        got = np.from_dlpack(
            drafter.draft(
                _tensor(block_hidden, DType.bfloat16),
                _tensor(anchors.astype(np.int64), DType.int64),
                _tensor(f_rnn, DType.bfloat16),
                _tensor(lm_head, DType.bfloat16),
                final_logit_softcapping=cap,
                unpadded_vocab_size=unpadded,
            ).to(CPU())
        )
    assert got.shape == (seqs, steps)

    # vLLM's _sample_sequential: base logits from a bf16 GEMM, hidden_out
    # folded into the LM head and rounded to bf16, and each step's logits
    # bf16(base + bias) softcapped. Teacher-forced on the drafted tokens, each
    # pick must be the reference's argmax up to a bf16 tie.
    rows = block_hidden.reshape(seqs, CONFIG.block_size, HIDDEN)[:, 1:]
    weight, bias = w["rnn_proj.weight"], w["rnn_proj.bias"]
    fold_weight = _bf16(lm_head @ w["hidden_out.weight"])
    fold_bias = _bf16(lm_head @ w["hidden_out.bias"])
    state = np.zeros((seqs, RANK), np.float32)
    previous = anchors
    for step in range(steps):
        hidden = rows[:, step]
        base = _bf16(hidden @ lm_head.T)
        inputs = np.concatenate(
            [state, w["markov_w1.weight"][previous], hidden, f_rnn], axis=-1
        )
        raw = _bf16(inputs @ weight.T + bias)
        gate, cand, out = np.split(raw, 3, axis=-1)
        gate = _bf16(1.0 / (1.0 + np.exp(-gate)))
        state = _bf16(
            _bf16(gate * state)
            + _bf16(_bf16(1.0 - gate) * _bf16(np.tanh(cand)))
        )
        latent = _bf16(np.tanh(out))
        step_bias = _bf16(latent @ fold_weight.T + fold_bias)
        logits = _bf16(cap * np.tanh(_bf16(base + step_bias) / cap))
        logits = logits[:, :unpadded]
        picked = got[:, step]
        best = logits.max(axis=-1)
        chosen = logits[np.arange(seqs), picked]
        np.testing.assert_array_less(best - chosen, 0.25 + 1e-6)
        previous = picked.astype(np.int64)
