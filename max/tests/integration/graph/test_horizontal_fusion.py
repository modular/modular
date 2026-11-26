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

"""End-to-end tests for the MAP-dialect fusion system's HorizontalFuser.

Sibling kernels over one index space that share an operand are merged into a
single multi-output kernel, so the shared operand is loaded once instead of
once per kernel. Unlike the other fusers there is no producer/consumer edge
here, which is why these check the fusion outcome and not only the numbers:
the unfused form computes the same answer, just with the duplicate load.

The pass also merges an elementwise producer whose result has other readers
with its consumer, which input fusion leaves alone. A consumer that reads a
strided window of the result runs only where the window selects the element.
Opaque kernels take no part in either merge.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from fusion_utils import execute, load_and_verify_fusion, run_and_verify_fusion
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


def _f32(*shape: int, device: DeviceRef | None = None) -> TensorType:
    return TensorType(
        DType.float32, list(shape), device=device or DeviceRef.CPU()
    )


def test_two_adds_share_operand(session: InferenceSession) -> None:
    r"""Two adds sharing an operand fuse into one two-output kernel.

    ```
      y   x   z            y   x   z
       \ / \ /              \  |  /
       add add     -->    [add, add]
    ```
    """
    with Graph(
        "two_adds_share_operand",
        input_types=[_f32(8), _f32(8), _f32(8)],
    ) as graph:
        x, y, z = (v.tensor for v in graph.inputs)
        graph.output(x + y, x + z)

    a = np.random.randn(8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    c = np.random.randn(8).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a, b, c, fused=r"mo\.add.*mo\.add"
    )
    np.testing.assert_allclose(out0, a + b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, a + c, rtol=1e-5, atol=1e-5)


def test_transitive_operand_sharing(session: InferenceSession) -> None:
    r"""Sharing need only be pairwise: no operand is common to all three adds,
    but fusing the first two leaves a kernel that shares an operand with the
    third, so the pass collapses the whole chain on a later round.

    ```
      x   y   z   w           x  y  z  w
       \ / \ / \ /             \ | | /
       add add add   -->   [add, add, add]
    ```
    """
    with Graph(
        "transitive_operand_sharing",
        input_types=[_f32(4), _f32(4), _f32(4), _f32(4)],
    ) as graph:
        x, y, z, w = (v.tensor for v in graph.inputs)
        graph.output(x + y, y + z, z + w)

    a = np.random.randn(4).astype(np.float32)
    b = np.random.randn(4).astype(np.float32)
    c = np.random.randn(4).astype(np.float32)
    d = np.random.randn(4).astype(np.float32)
    out0, out1, out2 = run_and_verify_fusion(
        session, graph, a, b, c, d, fused=r"mo\.add.*mo\.add.*mo\.add"
    )
    np.testing.assert_allclose(out0, a + b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, b + c, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out2, c + d, rtol=1e-5, atol=1e-5)


def test_matmul_feeds_two_adds(session: InferenceSession) -> None:
    """Horizontal fusion leaves a kernel output fusion can still absorb.

    The two adds share the matmul's result, so they merge first; only then
    does that result have the single use epilogue fusion requires, and the
    matmul takes the pair into its epilogue. The intermediate is never
    materialized -- the matmul's per-element value feeds both adds directly.
    """
    with Graph(
        "matmul_feeds_two_adds",
        input_types=[_f32(8, 8), _f32(8, 8), _f32(8, 8), _f32(8, 8)],
    ) as graph:
        a, b, y, z = (v.tensor for v in graph.inputs)
        mm = a @ b
        graph.output(mm + y, mm + z)

    a_np = np.random.randn(8, 8).astype(np.float32)
    b_np = np.random.randn(8, 8).astype(np.float32)
    y_np = np.random.randn(8, 8).astype(np.float32)
    z_np = np.random.randn(8, 8).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session,
        graph,
        a_np,
        b_np,
        y_np,
        z_np,
        fused=r"mo\.matmul.*mo\.add.*mo\.add",
    )
    expected = a_np @ b_np
    np.testing.assert_allclose(out0, expected + y_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out1, expected + z_np, rtol=1e-4, atol=1e-4)


def test_add_and_cast_share_operand(session: InferenceSession) -> None:
    r"""Two consumers of one input whose outputs differ in dtype.

    The fused kernel's destinations are no longer interchangeable: whether an
    accumulation buffer is needed is judged from a single epilogue result, so
    this pins that a dtype-changing sibling still stores its own dtype.

    ```
      y   x                y   x
       \ / \                \  |
       add  cast   -->   [add, cast]
       f32  f16           f32   f16
    ```
    """
    with Graph(
        "add_and_cast_share_operand",
        input_types=[_f32(8), _f32(8)],
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(x + y, ops.cast(x, DType.float16))

    a = np.random.randn(8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a, b, fused=r"mo\.add.*mo\.cast"
    )
    np.testing.assert_allclose(out0, a + b, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(out1, a.astype(np.float16))


def test_matmul_feeds_add_and_cast(session: InferenceSession) -> None:
    """The dtype-mixed pair, fused into a matmul's epilogue.

    `AllocateAccumulatingBuffers` decides whether the kernel needs a pre-cast
    accumulation buffer from `dtypeChanges && containsMatmulOrConvOrConcat`,
    reading the dtype off a single epilogue result -- so a matmul whose two
    fused destinations disagree is the case that exercises that choice.
    """
    with Graph(
        "matmul_feeds_add_and_cast",
        input_types=[_f32(8, 8), _f32(8, 8), _f32(8, 8)],
    ) as graph:
        a, b, y = (v.tensor for v in graph.inputs)
        mm = a @ b
        graph.output(mm + y, ops.cast(mm, DType.float16))

    a_np = np.random.randn(8, 8).astype(np.float32)
    b_np = np.random.randn(8, 8).astype(np.float32)
    y_np = np.random.randn(8, 8).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a_np, b_np, y_np, fused=r"mo\.matmul.*mo\.add.*mo\.cast"
    )
    expected = a_np @ b_np
    np.testing.assert_allclose(out0, expected + y_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out1.astype(np.float32),
        expected.astype(np.float16).astype(np.float32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_two_adds_share_operand_dynamic(session: InferenceSession) -> None:
    r"""The same pair over a symbolic extent.

    Fusion pairs kernels by iteration domain, which for a symbolic shape is an
    equality on the dim parameter rather than on a constant, so this covers the
    comparison the static case cannot.

    ```
      y[n] x[n] z[n]          y   x   z
         \ / \ /               \  |  /
         add add      -->    [add, add]
    ```
    """
    sym = TensorType(DType.float32, ["n"], device=DeviceRef.CPU())
    with Graph(
        "two_adds_share_operand_dynamic", input_types=[sym, sym, sym]
    ) as graph:
        x, y, z = (v.tensor for v in graph.inputs)
        graph.output(x + y, x + z)

    a = np.random.randn(12).astype(np.float32)
    b = np.random.randn(12).astype(np.float32)
    c = np.random.randn(12).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a, b, c, fused=r"mo\.add.*mo\.add"
    )
    np.testing.assert_allclose(out0, a + b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, a + c, rtol=1e-5, atol=1e-5)


def test_siblings_in_branch(session: InferenceSession) -> None:
    """Siblings inside a conditional's branch fuse there, in that branch.

    ```
      if pred:              if pred:
        x+y   x+z    -->      [x+y, x+z]
      else:                 else:
        x-y   x-z             [x-y, x-z]
    ```
    """
    pred_type = TensorType(DType.bool, [], device=DeviceRef.CPU())
    with Graph(
        "siblings_in_branch",
        input_types=[_f32(8), _f32(8), _f32(8), pred_type],
    ) as graph:
        x, y, z, pred = (v.tensor for v in graph.inputs)

        def then_fn():  # noqa: ANN202
            return [x + y, x + z]

        def else_fn():  # noqa: ANN202
            return [x - y, x - z]

        r0, r1 = ops.cond(pred, [_f32(8), _f32(8)], then_fn, else_fn)
        graph.output(r0, r1)

    model = session.load(graph)
    summaries = model.kernel_summaries
    assert any(re.search(r"mo\.add.*mo\.add", s) for s in summaries), summaries
    assert any(re.search(r"mo\.sub.*mo\.sub", s) for s in summaries), summaries

    a = np.random.randn(8).astype(np.float32)
    b = np.random.randn(8).astype(np.float32)
    c = np.random.randn(8).astype(np.float32)
    for pred_np, expected in (
        (np.array(True), (a + b, a + c)),
        (np.array(False), (a - b, a - c)),
    ):
        out0, out1 = execute(model, a, b, c, pred_np)
        np.testing.assert_allclose(out0, expected[0], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out1, expected[1], rtol=1e-5, atol=1e-5)


def test_reader_in_branch_between_siblings(session: InferenceSession) -> None:
    """A conditional that reads the first sibling, between the two, keeps them
    apart.

    The merged kernel would sit at the second sibling's position, below the
    conditional that reads its first result.

    ```
      s = x + y
      r = if pred: s * z  else: z      <- reads s
      t = x + w                        <- shares x with s, but stays apart
    ```
    """
    pred_type = TensorType(DType.bool, [], device=DeviceRef.CPU())
    with Graph(
        "reader_in_branch_between_siblings",
        input_types=[_f32(8), _f32(8), _f32(8), _f32(8), pred_type],
    ) as graph:
        x, y, z, w, pred = (v.tensor for v in graph.inputs)
        s = x + y

        def then_fn():  # noqa: ANN202
            return [s * z]

        def else_fn():  # noqa: ANN202
            return [z]

        (r,) = ops.cond(pred, [_f32(8)], then_fn, else_fn)
        t = x + w
        graph.output(t, r)

    model = session.load(graph)
    summaries = model.kernel_summaries
    assert not any(re.search(r"mo\.add.*mo\.add", s) for s in summaries), (
        summaries
    )

    a, b, c, d = (np.random.randn(8).astype(np.float32) for _ in range(4))
    for pred_np, branch in (
        (np.array(True), (a + b) * c),
        (np.array(False), c),
    ):
        out_t, out_r = execute(model, a, b, c, d, pred_np)
        np.testing.assert_allclose(out_t, a + d, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out_r, branch, rtol=1e-5, atol=1e-5)


def test_matmul_feeds_two_adds_dynamic(session: InferenceSession) -> None:
    """The matmul chain over a symbolic extent.

    `[N, N]` on every operand binds one dim parameter across the matmul, both
    adds and both outputs, so the horizontal pair and the epilogue edge are
    matched on a parameter rather than on a constant.
    """
    sym = TensorType(DType.float32, ["N", "N"], device=DeviceRef.CPU())
    with Graph(
        "matmul_feeds_two_adds_dynamic", input_types=[sym, sym, sym, sym]
    ) as graph:
        a, b, y, z = (v.tensor for v in graph.inputs)
        mm = a @ b
        graph.output(mm + y, mm + z)

    a_np = np.random.randn(6, 6).astype(np.float32)
    b_np = np.random.randn(6, 6).astype(np.float32)
    y_np = np.random.randn(6, 6).astype(np.float32)
    z_np = np.random.randn(6, 6).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session,
        graph,
        a_np,
        b_np,
        y_np,
        z_np,
        fused=r"mo\.matmul.*mo\.add.*mo\.add",
    )
    expected = a_np @ b_np
    np.testing.assert_allclose(out0, expected + y_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out1, expected + z_np, rtol=1e-4, atol=1e-4)


def test_gather_feeds_two_adds(session: InferenceSession) -> None:
    """The producer/epilogue chain through a store lambda.

    `mo.matmul` declares `FusedComputeOutputTensor`, so its epilogue is a
    compute lambda: it returns a value the kernel stores at its own index.
    `mo.gather` declares `FusedOutputTensor` instead, so the epilogue owns
    its stores. A multi-output epilogue has to issue one store per
    destination either way, which the matmul tests only cover on the
    compute side.
    """
    idx_type = TensorType(DType.int64, [4], device=DeviceRef.CPU())
    with Graph(
        "gather_feeds_two_adds",
        input_types=[_f32(6, 4), idx_type, _f32(4, 4), _f32(4, 4)],
    ) as graph:
        x, idx, y, z = (v.tensor for v in graph.inputs)
        g = ops.gather(x, idx, axis=0)
        graph.output(g + y, g + z)

    x_np = np.random.randn(6, 4).astype(np.float32)
    idx_np = np.array([0, 2, 5, 1], dtype=np.int64)
    y_np = np.random.randn(4, 4).astype(np.float32)
    z_np = np.random.randn(4, 4).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session,
        graph,
        x_np,
        idx_np,
        y_np,
        z_np,
        fused=r"mo\.gather.*mo\.add.*mo\.add",
    )
    expected = x_np[idx_np]
    np.testing.assert_allclose(out0, expected + y_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, expected + z_np, rtol=1e-5, atol=1e-5)


def test_matmul_result_read_in_branch(session: InferenceSession) -> None:
    """An escaping read inside a conditional whose result the consumer reads.

    The branches read the matmul's result without the `mo.if` listing it as
    an operand, so only they carry the edge back to the matmul. Merging the
    add into the matmul would have the matmul wait on the conditional that
    waits on it, so the add stays its own kernel.
    """
    pred_type = TensorType(DType.bool, [], device=DeviceRef.CPU())
    with Graph(
        "matmul_result_read_in_branch",
        input_types=[_f32(8, 8), _f32(8, 8), _f32(8, 8), pred_type],
    ) as graph:
        a, b, y, pred = (v.tensor for v in graph.inputs)
        mm = a @ b

        def then_fn():  # noqa: ANN202
            return [mm * y]

        def else_fn():  # noqa: ANN202
            return [mm - y]

        (r,) = ops.cond(pred, [_f32(8, 8)], then_fn, else_fn)
        graph.output(mm + r)

    a_np = np.random.randn(8, 8).astype(np.float32)
    b_np = np.random.randn(8, 8).astype(np.float32)
    y_np = np.random.randn(8, 8).astype(np.float32)
    mm_np = a_np @ b_np
    model = load_and_verify_fusion(session, graph)
    for pred_np, branch in (
        (np.array(True), mm_np * y_np),
        (np.array(False), mm_np - y_np),
    ):
        (out,) = execute(model, a_np, b_np, y_np, pred_np)
        np.testing.assert_allclose(out, mm_np + branch, rtol=1e-4, atol=1e-4)


_DEVICES = [
    pytest.param(DeviceRef.CPU(), id="cpu"),
    pytest.param(
        DeviceRef.GPU(),
        id="gpu",
        marks=pytest.mark.skipif(
            accelerator_count() == 0, reason="requires an accelerator"
        ),
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
def test_elementwise_result_escapes(
    session: InferenceSession, device: DeviceRef
) -> None:
    """An elementwise result read by one consumer and also returned.

    Input fusion can't take this: duplicating the add into the relu would not
    free its store, which the graph output still needs. So the pair merges
    into one kernel that stores the sum and hands the relu the value it just
    computed.
    """
    with Graph(
        "elementwise_result_escapes",
        input_types=[_f32(8, 8, device=device), _f32(8, 8, device=device)],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        x = a + b
        graph.output(x, ops.relu(x))

    a_np = np.random.randn(8, 8).astype(np.float32)
    b_np = np.random.randn(8, 8).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.add.*mo\.relu"
    )
    np.testing.assert_allclose(out0, a_np + b_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out1, np.maximum(a_np + b_np, 0), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_elementwise_result_escapes_two_consumers(
    session: InferenceSession, device: DeviceRef
) -> None:
    """Two consumers of an escaping elementwise result merge with each other
    first, then the pair joins the producer: one kernel, three outputs."""
    with Graph(
        "elementwise_result_escapes_two_consumers",
        input_types=[_f32(8, 8, device=device), _f32(8, 8, device=device)],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        x = a + b
        graph.output(x, ops.relu(x), ops.sin(x))

    a_np = np.random.randn(8, 8).astype(np.float32)
    b_np = np.random.randn(8, 8).astype(np.float32)
    out0, out1, out2 = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.add.*mo\.relu.*mo\.sin"
    )
    x_np = a_np + b_np
    np.testing.assert_allclose(out0, x_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, np.maximum(x_np, 0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out2, np.sin(x_np), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_elementwise_chain_escapes(
    session: InferenceSession, device: DeviceRef
) -> None:
    """A chain whose every link is also returned merges link by link."""
    with Graph(
        "elementwise_chain_escapes",
        input_types=[_f32(8, 8, device=device) for _ in range(4)],
    ) as graph:
        a, b, s, t = (v.tensor for v in graph.inputs)
        x = a + b
        y = x * s
        graph.output(x, y, y * t)

    a_np, b_np, s_np, t_np = (
        np.random.randn(8, 8).astype(np.float32) for _ in range(4)
    )
    out0, out1, out2 = run_and_verify_fusion(
        session,
        graph,
        a_np,
        b_np,
        s_np,
        t_np,
        fused=r"mo\.add.*mo\.mul.*mo\.mul",
    )
    x_np = a_np + b_np
    np.testing.assert_allclose(out0, x_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out1, x_np * s_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out2, x_np * s_np * t_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "window",
    [
        pytest.param((slice(None), slice(4, None)), id="window"),
        pytest.param((slice(None), slice(1, None, 2)), id="step"),
    ],
)
def test_elementwise_result_escapes_through_slice(
    session: InferenceSession,
    device: DeviceRef,
    window: tuple[slice, ...],
) -> None:
    """A returned elementwise result whose consumer reads a strided window of
    it: the consumer joins the producer's kernel and runs only where the
    window selects the element, storing at the window's index."""
    with Graph(
        "elementwise_result_escapes_through_slice",
        input_types=[
            _f32(4, 8, 16, device=device),
            _f32(4, 8, 16, device=device),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        x = a + b
        graph.output(x, ops.relu(x[window]))

    a_np = np.random.randn(4, 8, 16).astype(np.float32)
    b_np = np.random.randn(4, 8, 16).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.add.*mo\.slice.*mo\.relu"
    )
    x_np = a_np + b_np
    np.testing.assert_allclose(out0, x_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out1, np.maximum(x_np[window], 0), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_elementwise_result_escapes_through_slice_dynamic(
    session: InferenceSession, device: DeviceRef
) -> None:
    """The strided window over a symbolic outer dim: the select reads the
    runtime shapes the foreach kernel passes it."""
    sym = TensorType(DType.float32, ["n", 8, 16], device=device)
    with Graph(
        "elementwise_result_escapes_through_slice_dynamic",
        input_types=[sym, sym],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        x = a + b
        graph.output(x, ops.relu(x[:, 1::2]))

    a_np = np.random.randn(5, 8, 16).astype(np.float32)
    b_np = np.random.randn(5, 8, 16).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.add.*mo\.slice.*mo\.relu"
    )
    x_np = a_np + b_np
    np.testing.assert_allclose(out0, x_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out1, np.maximum(x_np[:, 1::2], 0), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_elementwise_result_escapes_through_slice_cast(
    session: InferenceSession, device: DeviceRef
) -> None:
    """A consumer that casts the strided window: the select stores a
    narrower dtype at the window's index."""
    with Graph(
        "elementwise_result_escapes_through_slice_cast",
        input_types=[
            _f32(4, 8, 16, device=device),
            _f32(4, 8, 16, device=device),
        ],
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        x = a + b
        graph.output(x, ops.cast(x[:, 1::2], DType.float16))

    a_np = np.random.randn(4, 8, 16).astype(np.float32)
    b_np = np.random.randn(4, 8, 16).astype(np.float32)
    out0, out1 = run_and_verify_fusion(
        session, graph, a_np, b_np, fused=r"mo\.add.*mo\.slice.*mo\.cast"
    )
    x_np = a_np + b_np
    np.testing.assert_allclose(out0, x_np, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        out1, x_np[:, 1::2].astype(np.float16), rtol=1e-3, atol=1e-3
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_matmul_pair_escapes_through_slice(
    session: InferenceSession, device: DeviceRef
) -> None:
    """Two adds of a matmul pair as siblings and a relu of a strided window of
    one joins them through a select; the whole kernel still moves into the
    matmul's epilogue, so the matmul's result is never stored."""
    with Graph(
        "matmul_pair_escapes_through_slice",
        input_types=[
            _f32(8, 8, device=device),
            _f32(8, 16, device=device),
            _f32(8, 16, device=device),
            _f32(8, 16, device=device),
        ],
    ) as graph:
        p, q, b, c = (v.tensor for v in graph.inputs)
        mm = p @ q
        x = mm + b
        graph.output(x, mm + c, ops.relu(x[1::2, :]))

    p_np, q_np = (np.random.randn(8, n).astype(np.float32) for n in (8, 16))
    b_np, c_np = (np.random.randn(8, 16).astype(np.float32) for _ in range(2))
    out0, out1, out2 = run_and_verify_fusion(
        session,
        graph,
        p_np,
        q_np,
        b_np,
        c_np,
        fused=r"mo\.matmul.*mo\.add.*mo\.add.*mo\.slice.*mo\.relu",
    )
    mm_np = p_np @ q_np
    np.testing.assert_allclose(out0, mm_np + b_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out1, mm_np + c_np, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out2, np.maximum(mm_np + b_np, 0)[1::2, :], rtol=1e-4, atol=1e-4
    )
