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

"""End-to-end tests for the fused KV-cache update: a concat returned whole
and as its trailing window, each in one kernel.

An attention block extends its k and v caches with the new rows and keeps the
last rows as the next window. These check that the concat and the window's
store share a kernel across the shapes the update takes: symbolic batch and
sequence extents, a GPU, bfloat16 caches, k and v together, and new rows
sliced out of a projection.
"""

from __future__ import annotations

import platform
import re

import numpy as np
import pytest
import torch
from fusion_utils import run_and_verify_fusion
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops

_CONCAT_SLICE = r"mo\.concat.*mo\.slice"

_BF16_DEVICES = [
    pytest.param(
        DeviceRef.CPU(),
        id="cpu",
        marks=pytest.mark.skipif(
            platform.machine() in ["arm64", "aarch64"],
            reason="BF16 is not supported on ARM CPU architecture",
        ),
    ),
    pytest.param(
        DeviceRef.GPU(),
        id="gpu",
        marks=pytest.mark.skipif(
            accelerator_count() == 0, reason="requires an accelerator"
        ),
    ),
]


def _run_bf16(
    session: InferenceSession, graph: Graph, *inputs: torch.Tensor, fused: str
) -> list[torch.Tensor]:
    """Runs ``graph`` like ``run_and_verify_fusion``, but on torch tensors,
    since numpy has no bfloat16."""
    model = session.load(graph)
    assert any(re.search(fused, s) for s in model.kernel_summaries), (
        model.kernel_summaries
    )
    results = model.execute(
        *(
            Buffer.from_dlpack(t).to(model.input_devices[i])
            for i, t in enumerate(inputs)
        )
    )
    return [torch.from_dlpack(r).cpu() for r in results]


def test_concat_slice_window_dynamic(session: InferenceSession) -> None:
    """The KV-cache window update, over symbolic batch and sequence extents.

    The window's start is negative on a dim only known at runtime, so the
    fused store resolves it against the concat's runtime shape.
    """
    cache_type = TensorType(
        DType.float32, ["batch", 8, 16], device=DeviceRef.CPU()
    )
    new_type = TensorType(
        DType.float32, ["batch", "seq", 16], device=DeviceRef.CPU()
    )
    with Graph(
        "concat_slice_window_dynamic", input_types=[cache_type, new_type]
    ) as graph:
        cache, new = (v.tensor for v in graph.inputs)
        full = ops.concat([cache, new], axis=1)
        graph.output(full, full[:, -8:])

    cache_np = np.random.randn(3, 8, 16).astype(np.float32)
    new_np = np.random.randn(3, 5, 16).astype(np.float32)
    full_out, window = run_and_verify_fusion(
        session, graph, cache_np, new_np, fused=_CONCAT_SLICE
    )
    expected = np.concatenate([cache_np, new_np], axis=1)
    np.testing.assert_array_equal(full_out, expected)
    np.testing.assert_array_equal(window, expected[:, -8:])


def test_concat_slice_kv_windows(session: InferenceSession) -> None:
    """The k and v cache updates of one attention block, each in one kernel.

    Both windows slice with the same bounds, so they share those inputs; the
    two slices must not pair with each other on that, or neither could fuse
    into its concat.
    """
    cache_type = TensorType(
        DType.float32, ["batch", 8, 16], device=DeviceRef.CPU()
    )
    new_type = TensorType(
        DType.float32, ["batch", "seq", 16], device=DeviceRef.CPU()
    )
    with Graph(
        "concat_slice_kv_windows",
        input_types=[cache_type, new_type, cache_type, new_type],
    ) as graph:
        k_cache, k_new, v_cache, v_new = (v.tensor for v in graph.inputs)
        k = ops.concat([k_cache, k_new], axis=1)
        v = ops.concat([v_cache, v_new], axis=1)
        graph.output(k, k[:, -8:], v, v[:, -8:])

    arrays = [
        np.random.randn(2, 8, 16).astype(np.float32),
        np.random.randn(2, 5, 16).astype(np.float32),
        np.random.randn(2, 8, 16).astype(np.float32),
        np.random.randn(2, 5, 16).astype(np.float32),
    ]
    model = session.load(graph)
    fused = [s for s in model.kernel_summaries if re.search(_CONCAT_SLICE, s)]
    assert len(fused) == 2, model.kernel_summaries
    k_out, k_win, v_out, v_win = run_and_verify_fusion(
        session, graph, *arrays, fused=_CONCAT_SLICE
    )
    k_expected = np.concatenate(arrays[:2], axis=1)
    v_expected = np.concatenate(arrays[2:], axis=1)
    np.testing.assert_array_equal(k_out, k_expected)
    np.testing.assert_array_equal(k_win, k_expected[:, -8:])
    np.testing.assert_array_equal(v_out, v_expected)
    np.testing.assert_array_equal(v_win, v_expected[:, -8:])


@pytest.mark.skipif(accelerator_count() == 0, reason="requires an accelerator")
def test_concat_slice_window_dynamic_gpu(session: InferenceSession) -> None:
    """The KV-cache window update on a GPU, where the fused store runs inside
    the concat kernel's device closure."""
    cache_type = TensorType(
        DType.float32, ["batch", 64, 4, 32], device=DeviceRef.GPU()
    )
    new_type = TensorType(
        DType.float32, ["batch", "seq", 4, 32], device=DeviceRef.GPU()
    )
    with Graph(
        "concat_slice_window_dynamic_gpu", input_types=[cache_type, new_type]
    ) as graph:
        cache, new = (v.tensor for v in graph.inputs)
        full = ops.concat([cache, new], axis=1)
        graph.output(full, full[:, -64:])

    cache_np = np.random.randn(2, 64, 4, 32).astype(np.float32)
    new_np = np.random.randn(2, 7, 4, 32).astype(np.float32)
    full_out, window = run_and_verify_fusion(
        session, graph, cache_np, new_np, fused=_CONCAT_SLICE
    )
    expected = np.concatenate([cache_np, new_np], axis=1)
    np.testing.assert_array_equal(full_out, expected)
    np.testing.assert_array_equal(window, expected[:, -64:])


@pytest.mark.parametrize("device", _BF16_DEVICES)
def test_concat_slice_window_bf16(
    session: InferenceSession, device: DeviceRef
) -> None:
    """The KV-cache window update in bfloat16, the dtype caches are kept in."""
    cache_type = TensorType(DType.bfloat16, ["batch", 8, 16], device=device)
    new_type = TensorType(DType.bfloat16, ["batch", "seq", 16], device=device)
    with Graph(
        "concat_slice_window_bf16", input_types=[cache_type, new_type]
    ) as graph:
        cache, new = (v.tensor for v in graph.inputs)
        full = ops.concat([cache, new], axis=1)
        graph.output(full, full[:, -8:])

    cache_t = torch.randn(3, 8, 16, dtype=torch.bfloat16)
    new_t = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    full_out, window = _run_bf16(
        session, graph, cache_t, new_t, fused=_CONCAT_SLICE
    )
    expected = torch.cat([cache_t, new_t], dim=1)
    torch.testing.assert_close(full_out, expected, rtol=0, atol=0)
    torch.testing.assert_close(window, expected[:, -8:], rtol=0, atol=0)


@pytest.mark.parametrize("device", _BF16_DEVICES)
def test_concat_slice_projection_window(
    session: InferenceSession, device: DeviceRef
) -> None:
    """A cache update whose new rows are sliced out of a projection.

    The new rows are one group of heads of the reshaped projection, as in an
    attention block's cache update. Prologue fusion reads them from the
    projection inside the concat, and the window's store joins the concat's
    epilogue, so the update and its window take one kernel.
    """
    cache_type = TensorType(DType.bfloat16, ["batch", 8, 2, 32], device=device)
    proj_type = TensorType(DType.bfloat16, ["batch", "seq", 192], device=device)
    with Graph(
        "concat_slice_projection_window", input_types=[cache_type, proj_type]
    ) as graph:
        cache, proj = (v.tensor for v in graph.inputs)
        heads = proj.reshape(["batch", "seq", 6, 32])
        full = ops.concat([cache, heads[:, :, 4:]], axis=1)
        graph.output(full, full[:, -8:])

    cache_t = torch.randn(2, 8, 2, 32, dtype=torch.bfloat16)
    proj_t = torch.randn(2, 3, 192, dtype=torch.bfloat16)
    fused = (
        r"Epilogue\(Prologue\(mo\.static\.reshape, mo\.slice, mo\.concat.*\), "
        r"mo\.slice"
    )
    full_out, window = _run_bf16(session, graph, cache_t, proj_t, fused=fused)
    new_rows = proj_t.reshape(2, 3, 6, 32)[:, :, 4:]
    expected = torch.cat([cache_t, new_rows], dim=1)
    torch.testing.assert_close(full_out, expected, rtol=0, atol=0)
    torch.testing.assert_close(window, expected[:, -8:], rtol=0, atol=0)
