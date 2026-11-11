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
"""The fused DSpark draft sampler against a float64 oracle.

The op adds the Markov bias to the head logits, scales by the per-row
temperature, draws one token per row by Gumbel-max over the full softmax and
returns that softmax as ``q``. The oracle rebuilds both in NumPy: ``q`` is
the tempered softmax of ``logits + bias``, and the drawn token is the
Gumbel-max the kernel's own Philox stream picks.

Rows are laid out at prime and ragged counts, at two temperatures. Vocabularies
cross the staged-slice budget so every launch shape runs: 2048 is one CTA per
row, 65536 splits a row over two CTAs and 131072 over four. A planted fault
checks that dropping the bias add changes the output.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from max.nn._graph_ops import _dspark_draft_sample

_PHILOX_SA = 0xD2511F53
_PHILOX_SB = 0xCD9E8D57
_PHILOX_10_0 = 0x9E3779B9
_PHILOX_10_1 = 0xBB67AE85
_UNIFORM_SCALE = np.float32(4.6566127342e-10)
_LOG2 = np.float32(0.6931471806)
_EPS = np.float32(1e-20)

# One CTA, two CTAs and four CTAs per row: the SM100 cluster kernel picks the
# narrowest cluster whose staged slice of the row fits shared memory.
_CASES = [
    (2048, 1),
    (2048, 5),
    (2048, 7),
    (2048, 13),
    (2048, 24),
    (65536, 3),
    (131072, 2),
]


def _mulhilow(a: int, b: int) -> tuple[int, int]:
    """The low and high halves of ``a * b`` as 32-bit words."""
    product = (a * b) & 0xFFFFFFFFFFFFFFFF
    return product & 0xFFFFFFFF, product >> 32


def _philox_round(
    counter: tuple[int, int, int, int], key: tuple[int, int]
) -> tuple[int, int, int, int]:
    """One round of the stdlib Philox mixer, ported from `philox.mojo`."""
    res0_lo, res0_hi = _mulhilow(_PHILOX_SA, counter[0])
    res1_lo, res1_hi = _mulhilow(_PHILOX_SB, counter[2])
    return (
        res1_hi ^ counter[1] ^ key[0],
        res1_lo,
        res0_hi ^ counter[3] ^ key[1],
        res0_lo,
    )


def _philox_step(seed: int) -> npt.NDArray[np.float32]:
    """The four uniforms `Random(seed=seed).step_uniform()` returns."""
    key0 = seed & 0xFFFFFFFF
    key1 = (seed >> 32) & 0xFFFFFFFF
    counter = (0, 0, 0, 0)
    for _ in range(9):
        counter = _philox_round(counter, (key0, key1))
        key0 = (key0 + _PHILOX_10_0) & 0xFFFFFFFF
        key1 = (key1 + _PHILOX_10_1) & 0xFFFFFFFF
    raw = _philox_round(counter, (key0, key1))
    words = np.array([word & 0x7FFFFFFF for word in raw], dtype=np.uint32)
    return words.astype(np.float32) * _UNIFORM_SCALE


def _gumbel_noise(seed: int, vocab: int) -> npt.NDArray[np.float32]:
    """The noise the kernel adds, keyed by the row seed and the chunk."""
    flat = np.concatenate(
        [_philox_step(seed * vocab + chunk) for chunk in range(vocab // 4)]
    )
    return -_LOG2 * np.log2(-np.log2(flat + _EPS) + _EPS)


def _expected(
    logits: npt.NDArray[np.float32],
    bias: npt.NDArray[np.float32],
    temperature: npt.NDArray[np.float32],
    seeds: npt.NDArray[np.uint64],
    vocab: int,
) -> tuple[
    npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.float32]
]:
    """The float64 q, the float32 Gumbel-max token, and the noised values."""
    x64 = (logits.astype(np.float64) + bias.astype(np.float64)) / (
        temperature.astype(np.float64)[:, None]
    )
    shifted = x64 - x64.max(axis=-1, keepdims=True)
    q = np.exp(shifted)
    q /= q.sum(axis=-1, keepdims=True)

    inv_temp = np.float32(1.0) / np.maximum(temperature, np.float32(1e-6))
    x32 = ((logits + bias) * inv_temp[:, None]).astype(np.float32)
    noised = x32 + np.stack([_gumbel_noise(int(seed), vocab) for seed in seeds])
    return noised.argmax(axis=-1).astype(np.int64), q, noised


@pytest.fixture(scope="module")
def draft_graph(session: InferenceSession) -> Model:
    device = DeviceRef.from_device(session.devices[0])
    with Graph(
        "dspark_draft_sample",
        input_types=[
            TensorType(DType.float32, ["rows", "vocab_size"], device=device),
            TensorType(DType.float32, ["rows", "vocab_size"], device=device),
            TensorType(DType.float32, ["rows"], device=device),
            TensorType(DType.uint64, ["rows"], device=device),
        ],
    ) as graph:
        logits, bias, temperature, seed = (v.tensor for v in graph.inputs)
        tokens, dist = _dspark_draft_sample(
            logits, bias, temperature=temperature, seed=seed
        )
        graph.output(tokens, dist)
    return session.load(graph)


def _run(
    model: Model,
    device: Accelerator,
    rows: int,
    vocab: int,
    temperature: float,
    seed_base: int,
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.float32],
    tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uint64],
    ],
]:
    rng = np.random.default_rng(seed_base)
    logits = (rng.standard_normal((rows, vocab)) * 2.0).astype(np.float32)
    bias = rng.standard_normal((rows, vocab)).astype(np.float32)
    temperatures = np.full(rows, temperature, np.float32)
    seeds = rng.integers(0, 2**63, rows, dtype=np.uint64)
    outputs = model(
        Buffer.from_numpy(logits).to(device),
        Buffer.from_numpy(bias).to(device),
        Buffer.from_numpy(temperatures).to(device),
        Buffer.from_numpy(seeds).to(device),
    )
    tokens = outputs[0].to_numpy()
    dist = outputs[1].to_numpy()
    return tokens, dist, (logits, bias, temperatures, seeds)


@pytest.mark.parametrize("vocab,rows", _CASES)
@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_fused_draft_sample_matches_oracle(
    draft_graph: Model,
    session: InferenceSession,
    vocab: int,
    rows: int,
    temperature: float,
) -> None:
    """q is the tempered softmax and the token is the oracle's Gumbel-max."""
    device = session.devices[0]
    assert isinstance(device, Accelerator)
    tokens, dist, (logits, bias, temperatures, seeds) = _run(
        draft_graph, device, rows, vocab, temperature, seed_base=rows * 31 + 7
    )
    expected_tokens, expected_q, noised = _expected(
        logits, bias, temperatures, seeds, vocab
    )

    assert dist.shape == (rows, vocab)
    assert dist.dtype == np.float32
    np.testing.assert_allclose(
        dist.astype(np.float64), expected_q, rtol=1e-5, atol=1e-6
    )
    # A row's probabilities are a normalized distribution.
    np.testing.assert_allclose(dist.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)

    # The token check is exact only where the oracle's winner is clear of the
    # runner-up by more than a rounding step; re-roll the rare near tie.
    sorted_noised = np.sort(noised, axis=-1)
    margin = sorted_noised[:, -1] - sorted_noised[:, -2]
    assert (margin > 1e-3).all(), "oracle near-tie; rerun with another seed"
    np.testing.assert_array_equal(tokens, expected_tokens)


def test_bias_is_used(draft_graph: Model, session: InferenceSession) -> None:
    """A planted fault: dropping the bias add must change the distribution."""
    device = session.devices[0]
    assert isinstance(device, Accelerator)
    vocab = 2048
    rows = 3
    _, dist, (logits, bias, temperatures, _) = _run(
        draft_graph, device, rows, vocab, 1.0, seed_base=101
    )
    x = logits.astype(np.float64) / temperatures.astype(np.float64)[:, None]
    shifted = x - x.max(axis=-1, keepdims=True)
    no_bias = np.exp(shifted)
    no_bias /= no_bias.sum(axis=-1, keepdims=True)
    assert np.abs(dist.astype(np.float64) - no_bias).max() > 1e-3
    assert np.abs(bias).max() > 1.0
