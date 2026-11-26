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
"""Checks the Inkling 2 DSpark drafter's head and mixing math on the host.

The references transcribe vLLM's ``markov_gate`` and ``mix_rmsnorm`` kernels
(``vllm/models/inkling2/nvidia/ops/dspark.py``) in NumPy, rounding to
bfloat16 where they do, on a config small enough to run eagerly.
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


def _numpy(x: Tensor) -> np.ndarray:
    return np.from_dlpack(x.cast(DType.float32).to(CPU()))


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


def test_the_markov_step_matches_the_reference_gate() -> None:
    rng = np.random.default_rng(0)
    head, w = _head(rng)
    batch = 4
    state = _bf16(rng.normal(size=(batch, RANK)).astype(np.float32))
    hidden = _bf16(rng.normal(size=(batch, HIDDEN)).astype(np.float32))
    f_rnn = _bf16(rng.normal(size=(batch, HIDDEN)).astype(np.float32))
    previous = rng.integers(0, VOCAB, size=(batch,))

    context = head.context(
        _tensor(hidden, DType.bfloat16), _tensor(f_rnn, DType.bfloat16)
    )
    new_state, latent = head.step(
        _tensor(state, DType.bfloat16),
        _tensor(previous.astype(np.int64), DType.int64),
        context,
    )

    # vLLM: the fp32 rnn_proj output rounds to bf16, then every torch op of
    # the gate rounds again.
    weight, bias = w["rnn_proj.weight"], w["rnn_proj.bias"]
    inputs = np.concatenate(
        [state, w["markov_w1.weight"][previous], hidden, f_rnn], axis=-1
    )
    raw = _bf16(inputs @ weight.T + bias)
    gate, cand, out = np.split(raw, 3, axis=-1)
    gate = _bf16(1.0 / (1.0 + np.exp(-gate)))
    cand = _bf16(np.tanh(cand))
    keep = _bf16(gate * state)
    update = _bf16(_bf16(1.0 - gate) * cand)
    expected_state = _bf16(keep + update)
    expected_latent = _bf16(np.tanh(out))

    np.testing.assert_allclose(
        _numpy(new_state), expected_state, rtol=2e-2, atol=2e-2
    )
    np.testing.assert_allclose(
        _numpy(latent), expected_latent, rtol=2e-2, atol=2e-2
    )


def test_the_feature_mix_matches_the_reference() -> None:
    rng = np.random.default_rng(1)
    rows = 5
    taps = _bf16(rng.normal(size=(rows, TAPS * HIDDEN)).astype(np.float32))
    fuse = _bf16(
        rng.normal(size=(CONFIG.num_fuse_rows, TAPS)).astype(np.float32)
    )
    norm_weight = _bf16(rng.normal(size=(HIDDEN,)).astype(np.float32))

    kv_params = construct_dspark_kv_params(
        CONFIG, KVCacheConfig(), [DeviceRef.CPU()], DType.bfloat16
    )
    with default_dtype(DType.bfloat16), default_device(CPU()):
        # Built lazily: only the tensors f_rnn reads are ever realized.
        with F.lazy():
            drafter = Inkling2DSpark(
                CONFIG, kv_params, max_seq_len=64, device=CPU()
            )
        weights = {
            name: np.zeros([int(d) for d in param.shape], np.float32)
            for name, param in drafter.parameters
        }
        weights["fuse_weight"] = fuse
        weights["markov_head.dflare_norm.weight"] = norm_weight
        drafter.load_state_dict(weights, auto_cast=True)
        got = _numpy(drafter.f_rnn(_tensor(taps, DType.bfloat16)))

    # vLLM's mix_rmsnorm over the Markov head's row (the last): the softmaxed
    # weights' fp32 sum, normalized and scaled in fp32, rounded once.
    alpha = np.exp(fuse[-1]) / np.exp(fuse[-1]).sum()
    mixed = (taps.reshape(rows, TAPS, HIDDEN) * alpha[None, :, None]).sum(
        axis=1
    )
    inv_rms = 1.0 / np.sqrt((mixed**2).mean(axis=-1, keepdims=True) + 1e-6)
    expected = _bf16(mixed * inv_rms * norm_weight)
    np.testing.assert_allclose(got, expected, rtol=1e-2, atol=1e-2)
