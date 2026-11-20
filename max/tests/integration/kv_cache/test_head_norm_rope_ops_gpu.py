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
"""Tests the per-head attention norm ops against the unfused ops, bit for
bit: the prologue's Q output and KV cache, and the epilogue's gated norm."""

from __future__ import annotations

import dataclasses

import ml_dtypes
import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn._graph_ops import rope_split_store_ragged
from max.nn.kv_cache import MHAKVCacheParams
from test_common.graph_utils import is_b100_b200
from test_common.simple_kv_cache import paged_kv_cache_inputs

HEAD_DIM = 64
N_Q = 16
N_KV = 2
PAGE_SIZE = 128
MAX_POS = 512
THETA = 5e5
EPS = 1e-5
Q_SCALE = 0.125
QKV_WIDTH = (N_Q + 2 * N_KV) * HEAD_DIM

sm100_only = pytest.mark.skipif(
    not is_b100_b200(),
    reason="the per-head attention norm ops run on B200 only",
)
RNG = np.random.default_rng(0)


def _bf16(arr: np.ndarray, device: Accelerator) -> Buffer:
    bits = np.ascontiguousarray(arr.astype(ml_dtypes.bfloat16).view(np.uint16))
    return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)


def _bits(buf: Buffer) -> np.ndarray:
    return buf.copy(device=CPU()).view(DType.uint16).to_numpy()


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=N_KV,
        head_dim=HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _rope_freqs() -> np.ndarray:
    """Interleaved ``[cos, sin]`` pairs, ``[MAX_POS, HEAD_DIM]``."""
    inv_freq = 1.0 / (THETA ** (np.arange(0, HEAD_DIM, 2) / HEAD_DIM))
    ang = np.outer(np.arange(MAX_POS), inv_freq)
    return np.stack([np.cos(ang), np.sin(ang)], axis=-1).reshape(
        MAX_POS, HEAD_DIM
    )


def _head_norm(x: TensorValue, n_heads: int) -> TensorValue:
    ones = ops.broadcast_to(
        ops.constant(1.0, DType.bfloat16, device=x.device), [HEAD_DIM]
    )
    normed = ops.rms_norm(
        x.reshape([-1, n_heads, HEAD_DIM]), ones, EPS, multiply_before_cast=True
    )
    return normed.reshape([-1, n_heads * HEAD_DIM])


def _prologue_graph(fused: bool) -> Graph:
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    kv_params = _kv_params()
    with Graph(
        "prologue",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", QKV_WIDTH], device=gpu),
            TensorType(DType.uint32, ["offsets"], device=gpu),
            TensorType(DType.bfloat16, [MAX_POS, HEAD_DIM], device=gpu),
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        qkv, offsets, freqs = (v.tensor for v in graph.inputs[:3])
        kv = kv_params.unflatten_kv_inputs(iter(graph.inputs[3:]))[0]
        layer = ops.constant(0, DType.uint32, device=cpu)
        if fused:
            q = ops.inplace_custom(
                "kv_cache.qk_rms_norm_rope_split_store.ragged.paged",
                device=gpu,
                values=[
                    qkv,
                    offsets,
                    freqs,
                    *kv.flatten_without_attention_dispatch_metadata(),
                    layer,
                    ops.constant(EPS, DType.float32, device=cpu),
                    ops.constant(Q_SCALE, DType.float32, device=cpu),
                ],
                out_types=[
                    TensorType(
                        DType.bfloat16, [qkv.shape[0], N_Q * HEAD_DIM], gpu
                    )
                ],
                parameters={"n_q_heads": N_Q},
            )[0].tensor
        else:
            q_dim, kv_dim = N_Q * HEAD_DIM, N_KV * HEAD_DIM
            x_q = qkv[:, :q_dim]
            x_k = qkv[:, q_dim : q_dim + kv_dim]
            x_v = qkv[:, q_dim + kv_dim :]
            x_q = _head_norm(x_q, N_Q) * Q_SCALE
            x_k = _head_norm(x_k, N_KV)
            q = rope_split_store_ragged(
                kv_params=kv_params,
                qkv=ops.concat([x_q, x_k, x_v], axis=-1),
                input_row_offsets=offsets,
                freqs_cis=freqs,
                kv_collection=kv,
                layer_idx=layer,
                n_heads=N_Q,
                interleaved=False,
            )
        graph.output(q)
    return graph


# Below and above B200's SM count: the op runs a different kernel in each.
@sm100_only
@pytest.mark.parametrize("prompt_lens", [[3, 5], [200, 120]])
def test_prologue_matches_unfused(prompt_lens: list[int]) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    tokens = sum(prompt_lens)
    qkv = RNG.standard_normal((tokens, QKV_WIDTH)) * 2.0
    offsets = np.concatenate([[0], np.cumsum(prompt_lens)]).astype(np.uint32)

    models = [session.load(_prologue_graph(fused)) for fused in (True, False)]
    results = []
    for model in models:
        kv = paged_kv_cache_inputs(_kv_params(), prompt_lens, total_num_pages=8)
        (q,) = model.execute(
            _bf16(qkv, device),
            Buffer.from_numpy(offsets).to(device),
            _bf16(_rope_freqs(), device),
            *tree.leaves(kv),
        )
        assert isinstance(q, Buffer)
        results.append((_bits(q), _bits(kv.kv_blocks)))
    (q_fused, kv_fused), (q_ref, kv_ref) = results
    np.testing.assert_array_equal(q_fused, q_ref)
    np.testing.assert_array_equal(kv_fused, kv_ref)


def _epilogue_graph(fused: bool) -> Graph:
    gpu = DeviceRef.GPU()
    width = N_Q * HEAD_DIM
    with Graph(
        "epilogue",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", width], device=gpu),
            TensorType(
                DType.bfloat16, ["tokens", QKV_WIDTH + width], device=gpu
            ),
        ],
    ) as graph:
        x, qkvg = (v.tensor for v in graph.inputs)
        if fused:
            out = ops.custom(
                "norm.head_rms_norm_sigmoid_gate",
                device=gpu,
                values=[
                    x,
                    qkvg,
                    ops.constant(EPS, DType.float32, device=DeviceRef.CPU()),
                ],
                out_types=[x.type],
                parameters={"n_heads": N_Q},
            )[0].tensor
        else:
            out = _head_norm(x, N_Q) * ops.sigmoid(qkvg[:, QKV_WIDTH:])
        graph.output(out)
    return graph


@sm100_only
@pytest.mark.parametrize("tokens", [8, 300])
def test_epilogue_matches_unfused(tokens: int) -> None:
    device = Accelerator()
    session = InferenceSession(devices=[device])
    models = [session.load(_epilogue_graph(fused)) for fused in (True, False)]
    x = _bf16(RNG.standard_normal((tokens, N_Q * HEAD_DIM)) * 2.0, device)
    qkvg = _bf16(
        RNG.standard_normal((tokens, QKV_WIDTH + N_Q * HEAD_DIM)), device
    )
    outs = []
    for model in models:
        (out,) = model.execute(x, qkvg)
        assert isinstance(out, Buffer)
        outs.append(_bits(out))
    np.testing.assert_array_equal(outs[0], outs[1])


G_HEAD_DIM = 128
G_Q_DIM = N_Q * G_HEAD_DIM
G_KV_DIM = N_KV * G_HEAD_DIM
# A trailing gate block, as in a fused [q | k | v | gate] projection.
G_QKV_WIDTH = 2 * G_Q_DIM + 2 * G_KV_DIM


def _raw_bits(buf: Buffer) -> np.ndarray:
    uint = DType.uint8 if buf.dtype.size_in_bytes == 1 else DType.uint16
    return buf.copy(device=CPU()).view(uint).to_numpy()


def _gamma_kv_params(dtype: DType) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=N_KV,
        head_dim=G_HEAD_DIM,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[DeviceRef.GPU()],
    )


def _gamma_prologue_graph(fused: bool, kv_dtype: DType) -> Graph:
    gpu, cpu = DeviceRef.GPU(), DeviceRef.CPU()
    kv_params = _gamma_kv_params(kv_dtype)
    with Graph(
        "gamma_prologue",
        input_types=[
            TensorType(DType.bfloat16, ["tokens", G_QKV_WIDTH], device=gpu),
            TensorType(DType.uint32, ["offsets"], device=gpu),
            TensorType(DType.float32, [MAX_POS, G_HEAD_DIM], device=gpu),
            TensorType(DType.bfloat16, [G_HEAD_DIM], device=gpu),
            TensorType(DType.bfloat16, [G_HEAD_DIM], device=gpu),
            *kv_params.flattened_kv_inputs(),
        ],
    ) as graph:
        qkv, offsets, freqs, q_gamma, k_gamma = (
            v.tensor for v in graph.inputs[:5]
        )
        kv = kv_params.unflatten_kv_inputs(iter(graph.inputs[5:]))[0]
        layer = ops.constant(0, DType.uint32, device=cpu)
        if fused:
            q = ops.inplace_custom(
                "kv_cache.qk_gamma_rms_norm_rope_split_store.ragged.paged",
                device=gpu,
                values=[
                    qkv,
                    offsets,
                    freqs,
                    q_gamma,
                    k_gamma,
                    *kv.flatten_without_attention_dispatch_metadata(),
                    layer,
                    ops.constant(EPS, DType.float32, device=cpu),
                ],
                out_types=[TensorType(kv_dtype, [qkv.shape[0], G_Q_DIM], gpu)],
                parameters={"n_q_heads": N_Q},
            )[0].tensor
        else:

            def norm(x: TensorValue, n: int, gamma: TensorValue) -> TensorValue:
                return ops.rms_norm(
                    x.reshape([-1, n, G_HEAD_DIM]),
                    gamma,
                    EPS,
                    multiply_before_cast=True,
                ).reshape([-1, n * G_HEAD_DIM])

            x_q = norm(qkv[:, :G_Q_DIM], N_Q, q_gamma)
            x_k = norm(qkv[:, G_Q_DIM : G_Q_DIM + G_KV_DIM], N_KV, k_gamma)
            x_v = qkv[:, G_Q_DIM + G_KV_DIM : G_Q_DIM + 2 * G_KV_DIM]
            q = rope_split_store_ragged(
                kv_params=kv_params,
                qkv=ops.concat([x_q, x_k, x_v], axis=-1),
                input_row_offsets=offsets,
                freqs_cis=freqs,
                kv_collection=kv,
                layer_idx=layer,
                n_heads=N_Q,
                interleaved=False,
                q_out_dtype=kv_dtype,
            )
        graph.output(q)
    return graph


@dataclasses.dataclass
class _GammaModels:
    device: Accelerator
    kv_dtype: DType
    fused: Model
    unfused: Model


# The graphs take every shape as an input, so each cache dtype compiles once.
@pytest.fixture(
    scope="module",
    params=[
        DType.bfloat16,
        pytest.param(
            DType.float8_e4m3fn,
            marks=pytest.mark.skipif(
                not is_b100_b200(), reason="Beam's FP8 KV cache runs on B200"
            ),
        ),
    ],
)
def gamma_models(request: pytest.FixtureRequest) -> _GammaModels:
    kv_dtype = request.param
    device = Accelerator()
    session = InferenceSession(devices=[device])
    return _GammaModels(
        device,
        kv_dtype,
        session.load(_gamma_prologue_graph(True, kv_dtype)),
        session.load(_gamma_prologue_graph(False, kv_dtype)),
    )


# A decode batch on cached positions, a chunked prefill across a page
# boundary, and prefills below and above B200's SM count.
@pytest.mark.parametrize(
    "prompt_lens, cache_lengths",
    [
        ([1], [300]),
        ([1, 1, 1], [5, 130, 400]),
        ([3, 5], [126, 257]),
        ([3, 5], None),
        ([200, 120], None),
    ],
)
def test_gamma_prologue_matches_unfused(
    gamma_models: _GammaModels,
    prompt_lens: list[int],
    cache_lengths: list[int] | None,
) -> None:
    device = gamma_models.device
    tokens = sum(prompt_lens)
    qkv = RNG.standard_normal((tokens, G_QKV_WIDTH)) * 2.0
    offsets = np.concatenate([[0], np.cumsum(prompt_lens)]).astype(np.uint32)
    inv_freq = 1.0 / (THETA ** (np.arange(0, G_HEAD_DIM, 2) / G_HEAD_DIM))
    ang = np.outer(np.arange(MAX_POS), inv_freq)
    freqs = np.stack([np.cos(ang), np.sin(ang)], axis=-1).reshape(
        MAX_POS, G_HEAD_DIM
    )
    q_gamma = RNG.uniform(0.5, 1.5, G_HEAD_DIM)
    k_gamma = RNG.uniform(0.5, 1.5, G_HEAD_DIM)

    results = []
    for model in (gamma_models.fused, gamma_models.unfused):
        kv = paged_kv_cache_inputs(
            _gamma_kv_params(gamma_models.kv_dtype),
            prompt_lens,
            cache_lengths=cache_lengths,
            total_num_pages=8,
        )
        (q,) = model.execute(
            _bf16(qkv, device),
            Buffer.from_numpy(offsets).to(device),
            Buffer.from_numpy(freqs.astype(np.float32)).to(device),
            _bf16(q_gamma, device),
            _bf16(k_gamma, device),
            *tree.leaves(kv),
        )
        assert isinstance(q, Buffer)
        results.append((_raw_bits(q), _raw_bits(kv.kv_blocks)))
    (q_fused, kv_fused), (q_ref, kv_ref) = results
    np.testing.assert_array_equal(q_fused, q_ref)
    np.testing.assert_array_equal(kv_fused, kv_ref)
