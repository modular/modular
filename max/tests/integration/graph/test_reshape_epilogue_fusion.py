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

"""End-to-end tests for reshape-into-epilogue fusion under advanced fusion.

Every graph here exercises
``MOToMAP`` + ``ReshapeEpilogueFuser`` end to end: a reshape between a
fused-output producer and a downstream elementwise op folds into the
producer's epilogue as a ``mogg.index.reshape`` store-index transform (the
store lands at the reshaped position), so the elementwise op fuses too and the
reshape no longer materializes its own kernel. These prove numeric correctness
across the reshape kinds -- the store-index math itself is pinned by
``max/kernels/test/graph_compiler/test_index_reshape.mojo`` and the IR shape by
the mo-opt suite. Distinct element values catch a scrambled store index, not
just a wrong output shape.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from fusion_utils import run_and_verify_fusion
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops


def test_matmul_reshape_add_static(session: InferenceSession) -> None:
    """`reshape(matmul(a, b), [4]) + bias`, all static: the flatten folds into
    the matmul's epilogue store index. Mirrors `matmul-reshape-add-epilogue-fusion.mlir`.
    """
    with Graph(
        "matmul_reshape_add_static",
        input_types=[
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [4], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, bias = (v.tensor for v in graph.inputs)
        graph.output(ops.reshape(a @ b, [4]) + bias)

    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.eye(2, dtype=np.float32)
    bias_np = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, bias_np, fused=r"mo\.matmul.*mo\.add"
    )
    np.testing.assert_allclose(
        out, (a_np @ b_np).reshape(4) + bias_np, rtol=1e-5, atol=1e-5
    )


def test_matmul_reshape_add_dynamic_batch(session: InferenceSession) -> None:
    """`reshape(matmul(a, b), [M, 2, 2]) + bias` with a dynamic OUTER (batch)
    dim `M`: it never enters a stride or a needed modulo, so the store-index
    transform needs no runtime shape. Mirrors
    `matmul-reshape-add-epilogue-fusion-dynamic.mlir`.
    """
    with Graph(
        "matmul_reshape_add_dynamic_batch",
        input_types=[
            TensorType(DType.float32, ["M", 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, 4], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["M", 2, 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, bias = (v.tensor for v in graph.inputs)
        mm = a @ b  # [M, 4]
        graph.output(ops.reshape(mm, [mm.shape[0], 2, 2]) + bias)

    a_np = np.ones((2, 2), dtype=np.float32)
    b_np = np.ones((2, 4), dtype=np.float32)
    bias_np = np.ones((2, 2, 2), dtype=np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, bias_np, fused=r"mo\.matmul.*mo\.add"
    )
    # ones @ ones (K=2) = 2 everywhere; reshape to [M, 2, 2]; + ones = 3.
    np.testing.assert_allclose(
        out, np.full((2, 2, 2), 3.0), rtol=1e-5, atol=1e-5
    )


def test_matmul_reshape_add_inner_dynamic(session: InferenceSession) -> None:
    """`reshape(matmul(a, b), [N, 2]) + bias` with an INNER dynamic dim `N`:
    `[2, N] -> [N, 2]` re-linearizes across `N`, so the transform reads the
    runtime shapes threaded into the epilogue functor. Mirrors
    `matmul-reshape-add-epilogue-fusion-inner-dynamic.mlir`.
    """
    with Graph(
        "matmul_reshape_add_inner_dynamic",
        input_types=[
            TensorType(DType.float32, [2, 2], device=DeviceRef.CPU()),
            TensorType(DType.float32, [2, "N"], device=DeviceRef.CPU()),
            TensorType(DType.float32, ["N", 2], device=DeviceRef.CPU()),
        ],
    ) as graph:
        a, b, bias = (v.tensor for v in graph.inputs)
        mm = a @ b  # [2, N]
        graph.output(ops.reshape(mm, [b.shape[1], 2]) + bias)

    a_np = np.ones((2, 2), dtype=np.float32)
    b_np = np.ones((2, 4), dtype=np.float32)
    bias_np = np.ones((4, 2), dtype=np.float32)
    (out,) = run_and_verify_fusion(
        session, graph, a_np, b_np, bias_np, fused=r"mo\.matmul.*mo\.add"
    )
    # ones @ ones (K=2) = 2 everywhere; reshape [2, 4] -> [4, 2]; + ones = 3.
    np.testing.assert_allclose(out, np.full((4, 2), 3.0), rtol=1e-5, atol=1e-5)


_DEVICES = [
    DeviceRef.CPU(),
    pytest.param(
        DeviceRef.GPU(),
        marks=pytest.mark.skipif(accelerator_count() == 0, reason="no GPU"),
    ),
]


# The matmul's summary lists its fused epilogue as `Epilogue(mo.matmul, ...)`,
# on CPU and GPU alike.
_ADD_IN_EPILOGUE = r"Epilogue\(mo\.matmul, [^)]*mo\.add"
_ADD_ON_ITS_OWN = r"^(?!.*Epilogue\(mo\.matmul).*mo\.add"


def _matmul_reshape_add(
    session: InferenceSession,
    device: DeviceRef,
    m: int,
    other_shape: list[int],
    build: Callable[[TensorValue, TensorValue], TensorValue],
    other_np: np.ndarray,
    reference: Callable[[np.ndarray, np.ndarray], np.ndarray],
    fused: bool = True,
) -> None:
    with Graph(
        "matmul_reshape_add",
        input_types=[
            TensorType(DType.float32, [m, 16], device=device),
            TensorType(DType.float32, [16, 8], device=device),
            TensorType(DType.float32, other_shape, device=device),
        ],
    ) as graph:
        a, b, other = (v.tensor for v in graph.inputs)
        graph.output(build(a @ b, other))

    a_np = np.random.randn(m, 16).astype(np.float32)
    b_np = np.random.randn(16, 8).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session,
        graph,
        a_np,
        b_np,
        other_np,
        fused=_ADD_IN_EPILOGUE if fused else _ADD_ON_ITS_OWN,
    )
    np.testing.assert_allclose(
        out, reference(a_np @ b_np, other_np), rtol=1e-4, atol=1e-4
    )


# The kernel calls its epilogue with vectors along its own output's innermost
# dim. Once the reshape moves onto the epilogue, the add's other operand is
# read at the reshaped index, stepping by the reshaped shape's innermost
# stride: right only if no vector crosses a row of that shape, or the operand
# is laid out row after row across it. Otherwise the reshape isn't fused.


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_reshape_innermost_split_broadcast_not_fused(
    session: InferenceSession, device: DeviceRef
) -> None:
    """`[3, 8] -> [4, 6]` plus a broadcast `[6]` bias: an 8-wide vector
    crosses rows of 6 and the bias row repeats, so the reshape stays its own
    kernel. Fused, this read the wrong bias elements."""
    _matmul_reshape_add(
        session,
        device,
        3,
        [6],
        lambda mm, x: ops.reshape(mm, [4, 6]) + ops.broadcast_to(x, [4, 6]),
        np.arange(1, 7, dtype=np.float32) * 100,
        lambda mm, x: mm.reshape(4, 6) + x,
        fused=False,
    )


def test_cpu_matmul_reshape_mul_transposed_not_fused(
    session: InferenceSession,
) -> None:
    """`transpose(m) * reshape(w @ x)`, `[8, 25] -> [8, 5, 5]`: the CPU
    matmul's 16-wide vectors cross rows of 5, and the transposed `m` isn't
    laid out row after row, so the reshape stays its own kernel. Fused, this
    read the wrong `m` elements (GEX-4597)."""
    device = DeviceRef.CPU()
    with Graph(
        "cpu_matmul_reshape_mul_transposed",
        input_types=[
            TensorType(DType.float32, [5, 5], device=device),
            TensorType(DType.float32, [8, 8], device=device),
            TensorType(DType.float32, [8, 25], device=device),
        ],
    ) as graph:
        m, w, x = (v.tensor for v in graph.inputs)
        graph.output(ops.transpose(m, 0, 1) * ops.reshape(w @ x, [8, 5, 5]))

    m_np = np.random.randn(5, 5).astype(np.float32)
    w_np = np.random.randn(8, 8).astype(np.float32)
    x_np = np.random.randn(8, 25).astype(np.float32)
    (out,) = run_and_verify_fusion(
        session,
        graph,
        m_np,
        w_np,
        x_np,
        fused=r"^(?!.*Epilogue\(mo\.matmul).*mo\.mul",
    )
    np.testing.assert_allclose(
        out, m_np.T * (w_np @ x_np).reshape(8, 5, 5), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_reshape_trailing_one_slice_not_fused(
    session: InferenceSession, device: DeviceRef
) -> None:
    """`[3, 8] -> [3, 8, 1]` plus `x[:, :, 0:1]` of `[3, 8, 4]`: the load
    steps by the slice's size-1 stride, 1, while its rows are 4 apart."""
    _matmul_reshape_add(
        session,
        device,
        3,
        [3, 8, 4],
        lambda mm, x: ops.reshape(mm, [3, 8, 1]) + x[:, :, 0:1],
        np.arange(96, dtype=np.float32).reshape(3, 8, 4),
        lambda mm, x: mm.reshape(3, 8, 1) + x[:, :, 0:1],
        fused=False,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_reshape_trailing_one_broadcast_fused(
    session: InferenceSession, device: DeviceRef
) -> None:
    """`[3, 8] -> [3, 8, 1]` plus a broadcast of `[3, 1, 1]`: both strides
    of the crossed pair are 0, so every lane reads the right element."""
    _matmul_reshape_add(
        session,
        device,
        3,
        [3, 1, 1],
        lambda mm, x: (
            ops.reshape(mm, [3, 8, 1]) + ops.broadcast_to(x, [3, 8, 1])
        ),
        np.arange(1, 4, dtype=np.float32).reshape(3, 1, 1) * 100,
        lambda mm, x: mm.reshape(3, 8, 1) + x,
        fused=True,
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_reshape_outer_split_broadcast_fused(
    session: InferenceSession, device: DeviceRef
) -> None:
    """`[6, 8] -> [2, 3, 8]` splits an outer dim: the innermost extent stays
    the kernel's, so no vector leaves a row, even of a broadcast."""
    _matmul_reshape_add(
        session,
        device,
        6,
        [2, 1, 8],
        lambda mm, x: (
            ops.reshape(mm, [2, 3, 8]) + ops.broadcast_to(x, [2, 3, 8])
        ),
        np.arange(16, dtype=np.float32).reshape(2, 1, 8) * 100,
        lambda mm, x: mm.reshape(2, 3, 8) + x,
        fused=True,
    )
