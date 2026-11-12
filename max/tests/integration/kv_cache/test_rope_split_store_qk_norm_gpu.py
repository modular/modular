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
"""``kv_cache.rope_split_store.ragged.paged.with_qk_norm`` against its decomposition.

The oracle is the same computation as separate ops: per-head ``rms_norm`` of
the Q, K and V regions (V without a gamma), a ``concat``, then the plain
``kv_cache.rope_split_store.ragged.paged``. Both graphs run the same norm and
RoPE kernels, so the roped Q and the stored K/V must match to rounding:
the two norm launches may pick different reduction tiers.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from max import tree
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn._graph_ops import rope_split_store_ragged
from max.nn.kv_cache import MHAKVCacheParams
from test_common.simple_kv_cache import paged_kv_cache_inputs

EPSILON = 1e-6
NUM_Q_HEADS = 4
PAGE_SIZE = 128
MAX_POS = 256
# A prefill request next to a decode step with history.
PROMPT_LENS = [3, 1]
CACHE_LENGTHS = [0, 7]
TOTAL_SEQ_LEN = sum(PROMPT_LENS)


def _kv_params(
    dtype: DType, n_kv_heads: int, head_dim: int, device_ref: DeviceRef
) -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        page_size=PAGE_SIZE,
        devices=[device_ref],
    )


def _to_device(arr: np.ndarray, dtype: DType, device: Device) -> Buffer:
    if dtype == DType.bfloat16:
        bits = np.ascontiguousarray(
            arr.astype(ml_dtypes.bfloat16).view(np.uint16)
        )
        return Buffer.from_numpy(bits).view(DType.bfloat16).to(device)
    return Buffer.from_numpy(np.ascontiguousarray(arr.astype(np.float32))).to(
        device
    )


def _from_device(buf: Buffer, dtype: DType) -> np.ndarray:
    out = buf.copy(device=CPU())
    if dtype == DType.bfloat16:
        return (
            out.view(DType.uint16)
            .to_numpy()
            .view(ml_dtypes.bfloat16)
            .astype(np.float32)
        )
    return out.to_numpy()


def _build_graph(
    dtype: DType,
    device_ref: DeviceRef,
    *,
    n_kv_heads: int,
    head_dim: int,
    k_eq_v: bool,
    interleaved: bool,
    fused: bool,
) -> Graph:
    kv_params = _kv_params(dtype, n_kv_heads, head_dim, device_ref)
    q_dim = NUM_Q_HEADS * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_dim = q_dim + kv_dim if k_eq_v else q_dim + 2 * kv_dim
    input_types = [
        TensorType(dtype, ["total_seq_len", qkv_dim], device=device_ref),
        TensorType(DType.uint32, ["input_row_offsets_len"], device=device_ref),
        TensorType(dtype, [MAX_POS, head_dim], device=device_ref),
        TensorType(dtype, [head_dim], device=device_ref),  # q_gamma
        TensorType(dtype, [head_dim], device=device_ref),  # k_gamma
        *kv_params.flattened_kv_inputs(),
    ]
    with Graph("rope_split_store_qk_norm", input_types=input_types) as g:
        qkv = g.inputs[0].tensor
        row_offsets = g.inputs[1].tensor
        freqs_cis = g.inputs[2].tensor
        q_gamma = g.inputs[3].tensor
        k_gamma = g.inputs[4].tensor
        kv_collection = kv_params.unflatten_kv_inputs(iter(g.inputs[5:]))[0]
        layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

        if fused:
            q = rope_split_store_ragged(
                kv_params,
                qkv=qkv,
                input_row_offsets=row_offsets,
                freqs_cis=freqs_cis,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=NUM_Q_HEADS,
                interleaved=interleaved,
                q_norm_weight=q_gamma,
                k_norm_weight=k_gamma,
                rms_norm_eps=EPSILON,
                k_eq_v=k_eq_v,
            )
        else:
            parts = ops.split(
                qkv,
                [q_dim, kv_dim] if k_eq_v else [q_dim, kv_dim, kv_dim],
                axis=-1,
            )
            ones = ops.broadcast_to(
                ops.constant(1.0, dtype, device=device_ref), [head_dim]
            )

            def norm(
                x: TensorValue, n_heads: int, gamma: TensorValue
            ) -> TensorValue:
                return ops.rms_norm(
                    x.reshape([-1, n_heads, head_dim]),
                    gamma,
                    EPSILON,
                    weight_offset=0.0,
                    multiply_before_cast=True,
                ).reshape([-1, n_heads * head_dim])

            normed = ops.concat(
                [
                    norm(parts[0], NUM_Q_HEADS, q_gamma),
                    norm(parts[1], n_kv_heads, k_gamma),
                    norm(parts[-1], n_kv_heads, ones),
                ],
                axis=-1,
            )
            q = rope_split_store_ragged(
                kv_params,
                qkv=normed,
                input_row_offsets=row_offsets,
                freqs_cis=freqs_cis,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=NUM_Q_HEADS,
                interleaved=interleaved,
            )
        g.output(q)
    return g


def _run(
    session: InferenceSession,
    device: Device,
    dtype: DType,
    *,
    n_kv_heads: int,
    head_dim: int,
    k_eq_v: bool,
    interleaved: bool,
    fused: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the roped Q and the KV blocks after one execute."""
    device_ref = DeviceRef.from_device(device)
    model = session.load(
        _build_graph(
            dtype,
            device_ref,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            k_eq_v=k_eq_v,
            interleaved=interleaved,
            fused=fused,
        )
    )
    kv_dim = n_kv_heads * head_dim
    qkv_dim = NUM_Q_HEADS * head_dim + kv_dim * (1 if k_eq_v else 2)
    rng = np.random.default_rng(seed=head_dim + n_kv_heads)
    qkv = rng.standard_normal((TOTAL_SEQ_LEN, qkv_dim)) * 2.0
    freqs = rng.uniform(-1.0, 1.0, (MAX_POS, head_dim))
    q_gamma = rng.uniform(0.5, 1.5, head_dim)
    k_gamma = rng.uniform(0.5, 1.5, head_dim)
    row_offsets = np.concatenate([[0], np.cumsum(PROMPT_LENS)]).astype(
        np.uint32
    )

    kv_rt = paged_kv_cache_inputs(
        _kv_params(dtype, n_kv_heads, head_dim, device_ref),
        PROMPT_LENS,
        cache_lengths=CACHE_LENGTHS,
    )
    (q,) = model.execute(
        _to_device(qkv, dtype, device),
        Buffer.from_numpy(row_offsets).to(device),
        _to_device(freqs, dtype, device),
        _to_device(q_gamma, dtype, device),
        _to_device(k_gamma, dtype, device),
        *tree.leaves(kv_rt),
    )
    assert isinstance(q, Buffer)
    return _from_device(q, dtype), _from_device(kv_rt.kv_blocks, dtype)


@pytest.mark.parametrize("device_kind", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "n_kv_heads, head_dim, k_eq_v",
    [(2, 256, False), (1, 512, True)],
    ids=["qkv", "k_eq_v"],
)
@pytest.mark.parametrize(
    "interleaved", [False, True], ids=["split", "interleaved"]
)
def test_with_qk_norm_matches_decomposition(
    device_kind: str,
    n_kv_heads: int,
    head_dim: int,
    k_eq_v: bool,
    interleaved: bool,
) -> None:
    if device_kind == "gpu":
        if accelerator_count() == 0:
            pytest.skip("needs a GPU")
        device: Device = Accelerator(0)
        dtype = DType.bfloat16
    else:
        device = CPU()
        dtype = DType.float32
    session = InferenceSession(devices=[device])

    def run(fused: bool) -> tuple[np.ndarray, np.ndarray]:
        return _run(
            session,
            device,
            dtype,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            k_eq_v=k_eq_v,
            interleaved=interleaved,
            fused=fused,
        )

    want_q, want_kv = run(fused=False)
    got_q, got_kv = run(fused=True)
    # One unit in the last place of the output dtype.
    tol = 2**-7 if dtype == DType.bfloat16 else 1e-5
    np.testing.assert_allclose(got_q, want_q, rtol=tol, atol=tol)
    np.testing.assert_allclose(got_kv, want_kv, rtol=tol, atol=tol)
    # The stores reached the cache: a no-op op would also match a no-op
    # oracle.
    assert np.any(want_kv != 0)
