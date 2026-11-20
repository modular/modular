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
"""GPU consumers for the sparse MLA prefill CPU/GPU split.

The graphs are compiled to MEFs by a CPU-only build action
(``:mla_sparse_prefill_mefs`` via ``mef_precompile.bzl``); these tests do NOT
compile. ``test_prefill_from_precompiled_mef`` initializes each
:data:`PREFILL_SPECS` MEF and drives a fresh prefill through the combined op's
sparse-prefill branch (``mla_sm100_prefill_sparse``), asserting shape + finite.
``test_prefill_only_matches_auto`` initializes the precompiled ``auto`` and
``prefill`` MEFs with shared weights and asserts they produce matching output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _mla_sparse_graphs import (
    HIDDEN_SIZE,
    MAX_BATCH_SIZE,
    PAGE_SIZE,
    PREFILL_LEN,
    PREFILL_SPECS,
    SPECS_BY_NAME,
    TOTAL_NUM_PAGES,
    PrefillSpec,
    make_multi_kv,
    weights_for,
)
from max import tree
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.pipelines.kv_cache import PagedKVCacheManager
from test_common.context_utils import create_text_context
from test_common.mef_precompile import init_from_mef, mefs_from_env
from torch.utils.dlpack import from_dlpack

# Unit-scale FP8 rounding of the latent cache, against the BF16 cache.
COSINE_FLOOR = 0.99


def _batch_context_length() -> Buffer:
    """Returns the host batch context length for one fresh prefill.

    Page-aligned like the model's batch processor computes it.
    """
    aligned = (PREFILL_LEN + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    return Buffer.from_numpy(np.array([aligned], dtype=np.int32))


@pytest.mark.parametrize("spec", PREFILL_SPECS, ids=lambda s: s.name)
def test_prefill_from_precompiled_mef(spec: PrefillSpec) -> None:
    """Init a CPU-precompiled MEF on GPU and run a fresh sparse prefill."""
    mef_path = mefs_from_env("MLA_MEF_RLOCATIONS")[f"{spec.name}.mef"]
    assert mef_path.is_file(), f"precompiled MEF missing: {mef_path}"

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])

    model = init_from_mef(session, mef_path, weights_for(spec))

    multi_kv = make_multi_kv(spec.num_heads, spec.cache_dtype)
    kv_manager = PagedKVCacheManager(
        params=multi_kv,
        total_num_pages=TOTAL_NUM_PAGES,
        session=session,
        max_batch_size=MAX_BATCH_SIZE,
    )
    context = create_text_context(np.empty(PREFILL_LEN))
    kv_manager.claim(context)
    kv_manager.alloc(context)
    kv_ri_pref = kv_manager.runtime_inputs([[context]])
    assert isinstance(kv_ri_pref, dict)

    t_pref = (
        torch.randn((PREFILL_LEN, HIDDEN_SIZE), dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    hidden_prefill = Buffer.from_dlpack(t_pref).to(device)
    row_prefill = Buffer.from_numpy(
        np.array([0, PREFILL_LEN], dtype=np.uint32)
    ).to(device)
    inputs = (
        hidden_prefill,
        row_prefill,
        *tree.leaves(kv_ri_pref),
        _batch_context_length(),
    )
    out_pref = model.execute(*inputs)[0]

    out_t = from_dlpack(out_pref).cpu()
    out_np = (
        out_t.float().numpy()
        if out_t.dtype == torch.bfloat16
        else out_t.numpy()
    )
    assert out_np.shape == (PREFILL_LEN, HIDDEN_SIZE)
    assert not np.isnan(out_np).any()
    assert np.all(np.isfinite(out_np))
    assert np.any(out_np != 0)

    (captured,) = model.capture(1, *inputs)
    model.replay(1, *inputs)
    replay_np = from_dlpack(captured).cpu().float().numpy()
    np.testing.assert_array_equal(replay_np, out_np)
    model.replay(1, *inputs)
    np.testing.assert_array_equal(
        from_dlpack(captured).cpu().float().numpy(), out_np
    )


@pytest.mark.parametrize("num_heads", [128, 8])
def test_prefill_only_matches_auto(num_heads: int) -> None:
    """Disaggregated ``prefill_only`` routes through the same combined op as ``auto``.

    Initializes the CPU-precompiled ``auto`` and ``prefill`` MEFs (same head
    count) with shared weights, runs the same fresh prefill through each, and
    asserts the outputs match within a tight tolerance -- the only expected
    delta is nondeterministic reduction order across two kernel launches. For a
    pure prefill batch both modes assemble the identical graph, so a larger
    divergence would mean the prefill branch isn't reaching
    ``mla_sm100_prefill_sparse``. Runs at 128 heads (DeepSeek V3.2) and the GLM
    5.2 TP8-sharded count of 8 heads per device.
    """
    mefs = mefs_from_env("MLA_MEF_RLOCATIONS")
    # Shared weights across both modes, so a divergence isolates the graph-mode
    # routing rather than a weight mismatch (weights are mode-independent).
    weights = weights_for(SPECS_BY_NAME[f"prefill_fp8_{num_heads}h"])
    multi_kv = make_multi_kv(num_heads)

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])

    t_pref = (
        torch.randn((PREFILL_LEN, HIDDEN_SIZE), dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    hidden_prefill = Buffer.from_dlpack(t_pref).to(device)
    row_prefill = Buffer.from_numpy(
        np.array([0, PREFILL_LEN], dtype=np.uint32)
    ).to(device)

    def run(mef_name: str) -> np.ndarray:
        model = init_from_mef(session, mefs[mef_name], weights)
        # A fresh manager keeps each run's prefill cache writes independent.
        kv_manager = PagedKVCacheManager(
            params=multi_kv,
            total_num_pages=TOTAL_NUM_PAGES,
            session=session,
            max_batch_size=MAX_BATCH_SIZE,
        )
        context = create_text_context(np.empty(PREFILL_LEN))
        kv_manager.claim(context)
        kv_manager.alloc(context)
        kv_ri = kv_manager.runtime_inputs([[context]])
        assert isinstance(kv_ri, dict)
        out_buf = model.execute(
            hidden_prefill,
            row_prefill,
            *tree.leaves(kv_ri),
            _batch_context_length(),
        )[0]
        out_t = from_dlpack(out_buf).cpu()
        return (
            out_t.float().numpy()
            if out_t.dtype == torch.bfloat16
            else out_t.numpy()
        )

    out_auto = run(f"prefill_fp8_{num_heads}h.mef")
    out_prefill = run(f"prefill_only_fp8_{num_heads}h.mef")

    assert out_prefill.shape == (PREFILL_LEN, HIDDEN_SIZE)
    assert not np.isnan(out_prefill).any()
    assert np.all(np.isfinite(out_prefill))
    np.testing.assert_allclose(out_prefill, out_auto, atol=1e-3, rtol=1e-2)


@pytest.mark.parametrize(
    "fp8kv_name",
    [
        spec.name
        for spec in PREFILL_SPECS
        if spec.cache_dtype == DType.float8_e4m3fn
        and spec.name.replace("_fp8kv", "") in SPECS_BY_NAME
    ],
)
def test_fp8_cache_prefill_matches_bf16_cache(fp8kv_name: str) -> None:
    """Prefill over a unit-scale FP8 cache tracks the same prefill over BF16.

    Both graphs share weights and input; they differ only in the cache dtype,
    so the FP8-cache output (sparse decode kernel) must stay within FP8
    rounding of the BF16-cache output (BF16 sparse prefill kernel).
    """
    mefs = mefs_from_env("MLA_MEF_RLOCATIONS")
    fp8kv = SPECS_BY_NAME[fp8kv_name]
    bf16kv = SPECS_BY_NAME[fp8kv_name.replace("_fp8kv", "")]
    weights = weights_for(bf16kv)

    device = Accelerator(0)
    session = InferenceSession(devices=[Accelerator()])
    t_pref = (
        torch.randn((PREFILL_LEN, HIDDEN_SIZE), dtype=torch.float32) * 0.02
    ).to(torch.bfloat16)
    hidden_prefill = Buffer.from_dlpack(t_pref).to(device)
    row_prefill = Buffer.from_numpy(
        np.array([0, PREFILL_LEN], dtype=np.uint32)
    ).to(device)

    def run(spec: PrefillSpec) -> np.ndarray:
        model = init_from_mef(session, mefs[f"{spec.name}.mef"], weights)
        kv_manager = PagedKVCacheManager(
            params=make_multi_kv(spec.num_heads, spec.cache_dtype),
            total_num_pages=TOTAL_NUM_PAGES,
            session=session,
            max_batch_size=MAX_BATCH_SIZE,
        )
        context = create_text_context(np.empty(PREFILL_LEN))
        kv_manager.claim(context)
        kv_manager.alloc(context)
        kv_ri = kv_manager.runtime_inputs([[context]])
        assert isinstance(kv_ri, dict)
        out_buf = model.execute(
            hidden_prefill,
            row_prefill,
            *tree.leaves(kv_ri),
            _batch_context_length(),
        )[0]
        return from_dlpack(out_buf).cpu().float().numpy()

    want = run(bf16kv)
    got = run(fp8kv)
    assert np.any(want != 0)
    cosine = float(
        np.dot(got.ravel(), want.ravel())
        / (np.linalg.norm(got) * np.linalg.norm(want))
    )
    print(f"{fp8kv_name}: cosine={cosine:.6f}")
    assert cosine >= COSINE_FLOOR, cosine
