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

"""End-to-end tests for the new MAP-dialect fusion system's view handling.

Every graph here exercises
``MOToMAP``'s per-view-kind lowering (slice/broadcast/transpose/reshape) and
``ViewFuser`` end to end. `test_transpose_of_add_fuses_with_correct_index` in
particular is the Python-level regression test for the `ElementwiseFuser`
fix: a same-shape, non-identity view must be fused by `ViewFuser` (composing
index transforms correctly), never by `ElementwiseFuser` (which would splice
it in at the wrong coordinate). Whether the graph actually fused into one
kernel is covered by that MLIR test suite, not here -- these are e2e
correctness tests, checked against numpy as the oracle.

Every graph is built into one `Module` at import and compiled by a single
`load_all`: each compile has a fixed cost of about 10 s, which is most of the
cost of these small graphs.
"""

from __future__ import annotations

import numpy as np
import pytest
from fusion_utils import check_fusion, execute
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Module, TensorType, ops

_MODULE = Module()


def _cpu(dtype: DType, shape: list[int | str]) -> TensorType:
    return TensorType(dtype, shape, device=DeviceRef.CPU())


def _graph(name: str, *inputs: list[int | str] | TensorType) -> Graph:
    """A CPU graph in the shared module; a bare shape means float32."""
    types = [
        i if isinstance(i, TensorType) else _cpu(DType.float32, i)
        for i in inputs
    ]
    return Graph(name, input_types=types, module=_MODULE)


@pytest.fixture(scope="module")
def models(session: InferenceSession) -> dict[str, Model]:
    return session.load_all(_MODULE)


with _graph("slice_static", [8, 8]) as slice_static:
    (x,) = (v.tensor for v in slice_static.inputs)
    slice_static.output(x[2:6, 2:6])


def test_slice_static(models: dict[str, Model]) -> None:
    """A standalone `mo.slice` with static bounds compiles and runs correctly."""
    a = np.random.randn(8, 8).astype(np.float32)
    (out,) = execute(models[slice_static.name], a)
    np.testing.assert_allclose(out, a[2:6, 2:6], rtol=1e-5, atol=1e-5)


with _graph("slice_dynamic", ["n", 8]) as slice_dynamic:
    (x,) = (v.tensor for v in slice_dynamic.inputs)
    slice_dynamic.output(x[2:6, 2:6])


def test_slice_dynamic(models: dict[str, Model]) -> None:
    """A `mo.slice` over a symbolic input shape compiles and runs correctly."""
    a = np.random.randn(11, 8).astype(np.float32)
    (out,) = execute(models[slice_dynamic.name], a)
    np.testing.assert_allclose(out, a[2:6, 2:6], rtol=1e-5, atol=1e-5)


with _graph("broadcast_rank_preserving", [1, 4]) as broadcast_rank_preserving:
    (x,) = (v.tensor for v in broadcast_rank_preserving.inputs)
    broadcast_rank_preserving.output(x.broadcast_to([3, 4]))


def test_broadcast_rank_preserving(models: dict[str, Model]) -> None:
    """A rank-preserving `mo.static.broadcast_to` compiles and runs correctly."""
    a = np.random.randn(1, 4).astype(np.float32)
    (out,) = execute(models[broadcast_rank_preserving.name], a)
    np.testing.assert_allclose(
        out, np.broadcast_to(a, (3, 4)), rtol=1e-5, atol=1e-5
    )


with _graph("broadcast_rank_expanding", [4]) as broadcast_rank_expanding:
    (x,) = (v.tensor for v in broadcast_rank_expanding.inputs)
    broadcast_rank_expanding.output(x.broadcast_to([3, 4]))


def test_broadcast_rank_expanding(models: dict[str, Model]) -> None:
    """A rank-expanding `mo.static.broadcast_to` compiles and runs correctly."""
    a = np.random.randn(4).astype(np.float32)
    (out,) = execute(models[broadcast_rank_expanding.name], a)
    np.testing.assert_allclose(
        out, np.broadcast_to(a, (3, 4)), rtol=1e-5, atol=1e-5
    )


with _graph(
    "slice_of_dynamic_broadcast", ["n", 1], ["n", 4]
) as slice_of_dynamic_broadcast:
    x, y = (v.tensor for v in slice_of_dynamic_broadcast.inputs)
    slice_of_dynamic_broadcast.output((x.broadcast_to([x.shape[0], 4]) + y)[1:])


def test_slice_of_dynamic_broadcast(models: dict[str, Model]) -> None:
    """A slice over a broadcast with a symbolic dim reads the right rows.

    The slice clamps its bounds against the broadcast view's runtime shape, so
    that shape has to carry the symbolic dim's value, not a placeholder.
    """
    model = models[slice_of_dynamic_broadcast.name]
    check_fusion(model)
    a = np.random.randn(6, 1).astype(np.float32)
    b = np.random.randn(6, 4).astype(np.float32)
    (out,) = execute(model, a, b)
    np.testing.assert_allclose(out, (a + b)[1:], rtol=1e-5, atol=1e-5)


with _graph("transpose_static", [2, 3]) as transpose_static:
    (x,) = (v.tensor for v in transpose_static.inputs)
    transpose_static.output(ops.transpose(x, 0, 1))


def test_transpose_static(models: dict[str, Model]) -> None:
    """A standalone `mo.transpose` (2D swap) compiles and runs correctly."""
    a = np.random.randn(2, 3).astype(np.float32)
    (out,) = execute(models[transpose_static.name], a)
    np.testing.assert_allclose(out, a.T, rtol=1e-5, atol=1e-5)


_PERMUTE_DTYPES = [DType.uint8, DType.float32]
_PERMUTE_CHANNELS = [2, 3, 4, 5, 8]
_PERMUTE_GRAPHS: dict[tuple[DType, int], Graph] = {}

for _dtype in _PERMUTE_DTYPES:
    for _channels in _PERMUTE_CHANNELS:
        with _graph(
            f"permute_nhwc_to_nchw_{_dtype.name}_{_channels}",
            _cpu(_dtype, [1, 5, 7, _channels]),
        ) as permute:
            (x,) = (v.tensor for v in permute.inputs)
            permute.output(x.permute([0, 3, 1, 2]))
        _PERMUTE_GRAPHS[_dtype, _channels] = permute


@pytest.mark.parametrize("dtype", _PERMUTE_DTYPES)
@pytest.mark.parametrize("channels", _PERMUTE_CHANNELS)
def test_permute_nhwc_to_nchw(
    models: dict[str, Model], dtype: DType, channels: int
) -> None:
    """An NHWC -> NCHW permute reads its input with an innermost stride of
    `channels`, the case small-stride window loads target on CPU."""
    shape = [1, 5, 7, channels]
    a = (
        (np.arange(np.prod(shape)) * 7 % 251)
        .astype(dtype.to_numpy())
        .reshape(shape)
    )
    (out,) = execute(models[_PERMUTE_GRAPHS[dtype, channels].name], a)
    np.testing.assert_array_equal(out, a.transpose(0, 3, 1, 2))


with _graph("reshape_static", [4, 6]) as reshape_static:
    (x,) = (v.tensor for v in reshape_static.inputs)
    reshape_static.output(x.reshape([24]))


def test_reshape_static(models: dict[str, Model]) -> None:
    """A standalone `mo.static.reshape` (flatten) compiles and runs correctly."""
    a = np.random.randn(4, 6).astype(np.float32)
    (out,) = execute(models[reshape_static.name], a)
    np.testing.assert_allclose(out, a.reshape([24]), rtol=1e-5, atol=1e-5)


with _graph("chained_slice", [10, 10]) as chained_slice:
    (x,) = (v.tensor for v in chained_slice.inputs)
    chained_slice.output(x[1:9, 1:9][1:7, 1:7])


def test_chained_slice_fuses(models: dict[str, Model]) -> None:
    """A slice of a slice fuses into one kernel, composing against the first
    slice's view rather than the original tensor.
    """
    model = models[chained_slice.name]
    check_fusion(model, r"mo\.slice.*mo\.slice")
    a = np.random.randn(10, 10).astype(np.float32)
    (out,) = execute(model, a)
    np.testing.assert_allclose(out, a[1:9, 1:9][1:7, 1:7], rtol=1e-5, atol=1e-5)


with _graph(
    "view_into_prologue", [10], _cpu(DType.int64, [3])
) as view_into_prologue:
    x, indices = (v.tensor for v in view_into_prologue.inputs)
    view_into_prologue.output(ops.gather(x[2:8], indices, axis=0))


def test_view_fuses_into_existing_prologue(models: dict[str, Model]) -> None:
    """A slice fuses into `mo.gather`'s existing prologue capture.

    Regression test for the `LoopInvariantViewMotion` fix that let it hoist
    an index-transform chain out of a `map.iter.prologue`, not just a
    `map.iter.elementwise.body` (see
    GraphCompiler/test/mo-opt/MAPDialect/Transforms/LoopInvariantViewMotion/
    loop-invariant-view-motion-prologue.mlir), and for the `EmitMojo` fix
    letting a fused prologue/epilogue capture a view (not just a plain
    materialized tensor) -- `collectFusedCaptureInfos`
    (GraphCompiler/lib/MOGGDialect/MOGGOpsEmitters.cpp) now reads a capture's layout
    off its own already-emitted `ManagedTensorSlice.to_tile_tensor()` instead
    of assuming it's always a plain `mogg.tensor.create` result.
    """
    model = models[view_into_prologue.name]
    check_fusion(model, r"mo\.slice.*mo\.gather")
    a = np.random.randn(10).astype(np.float32)
    idx = np.array([0, 3, 5], dtype=np.int64)
    (out,) = execute(model, a, idx)
    np.testing.assert_allclose(out, a[2:8][idx], rtol=1e-5, atol=1e-5)


with _graph("broadcast_into_epilogue", [4, 4], [4]) as broadcast_into_epilogue:
    x, bias = (v.tensor for v in broadcast_into_epilogue.inputs)
    reduced = ops.max(x, axis=0)
    broadcast_into_epilogue.output(reduced + bias.broadcast_to([1, 4]))


def test_broadcast_fuses_into_existing_epilogue(
    models: dict[str, Model],
) -> None:
    """A broadcast bias fuses into `mo.reduce.max`'s existing epilogue capture.

    Regression test for `ViewFuser`'s epilogue generalization (mirrors
    `broadcastIntoExistingEpilogue` in
    GraphCompiler/test/mo-opt/MAPDialect/Transforms/ViewFusion/
    view_fusion_into_epilogue.mlir) and for the `EmitMojo` fused-capture fix
    (`collectFusedCaptureInfos` in GraphCompiler/lib/MOGGDialect/MOGGOpsEmitters.cpp)
    reading a capture's layout off its own `ManagedTensorSlice.to_tile_tensor()`
    instead of assuming every capture is a plain `mogg.tensor.create` result:
    `EpilogueFuser` first fuses the bias-add into the reduction's epilogue,
    capturing the broadcast's own result; `ViewFuser` then splices the
    broadcast itself into that same capture, since a transform op's
    `ParentOneOf` now permits living inside an epilogue.

    Uses `ops.max` rather than `ops.matmul` deliberately: a matmul-fused
    epilogue binds through `_bind_to_fused_compute_output` (a "compute
    lambda"), a separate code path, while `ops.max`'s epilogue binds through
    the plain `_bind_to_fused_output` ("store lambda") path this fix actually
    covers, keeping this test isolated to what `ViewFuser`'s generalization is
    responsible for.
    """
    model = models[broadcast_into_epilogue.name]
    check_fusion(model, r"mo\.reduce\.max.*mo\.add")
    x_np = np.random.randn(4, 4).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)
    (out,) = execute(model, x_np, bias_np)
    ref = np.max(x_np, axis=0, keepdims=True) + bias_np
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


with _graph("view_fanout", [8]) as view_fanout:
    (x,) = (v.tensor for v in view_fanout.inputs)
    y = x[1:5]
    view_fanout.output(ops.relu(y) + ops.negate(y))


def test_view_fanout_duplicates(models: dict[str, Model]) -> None:
    """A view feeding two different elementwise consumers fuses into both.

    Unlike `ElementwiseFuser`'s cost-gated duplication, `ViewFuser` always
    considers a view cheap to duplicate.
    """
    model = models[view_fanout.name]
    check_fusion(model, r"mo\.slice.*mo\.relu")
    a = np.random.randn(8).astype(np.float32)
    (out,) = execute(model, a)
    y_ref = a[1:5]
    ref = np.maximum(y_ref, 0) + (-y_ref)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


with _graph(
    "reshape_into_prologue", [10, 64], _cpu(DType.int64, [3])
) as reshape_into_prologue:
    x, indices = (v.tensor for v in reshape_into_prologue.inputs)
    chain = ops.transpose(x[1:4, 32:64].reshape([16, 6]), 0, 1)
    reshape_into_prologue.output(ops.gather(chain, indices, axis=0))


def test_reshape_and_view_fuse_into_opaque_prologue(
    models: dict[str, Model],
) -> None:
    """A reshape (and the transpose below it) fuse into an opaque's prologue;
    the view feeding the reshape stays materialized.

    `gather` is the opaque with a prologue; its input is
    `transpose(reshape(slice(x)))`. A reshape changes rank, so
    `LoopInvariantViewMotion` can't hoist a view fused atop a reshaped load into
    a stride view -- so the reshape is the terminal (most-upstream) fused view:
    the reshape and the transpose below it compose into gather's prologue, while
    the slice feeding the reshape stays its own materialized kernel. Mirrors
    `sliceReshapeTransposeIntoPrologue` in
    GraphCompiler/test/mo-opt/MAPDialect/Transforms/ViewFusion/
    view_fusion_reshape_barrier.mlir. An e2e correctness check (compiles under
    the reshape-in-prologue lowering, matches numpy); the fusion structure is
    covered by that lit test.
    """
    a = np.random.randn(10, 64).astype(np.float32)
    idx = np.array([0, 3, 5], dtype=np.int64)
    (out,) = execute(models[reshape_into_prologue.name], a, idx)
    ref = a[1:4, 32:64].reshape([16, 6]).T[idx]
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


with _graph("reshape_into_epilogue", [4, 4], [4]) as reshape_into_epilogue:
    x, bias = (v.tensor for v in reshape_into_epilogue.inputs)
    reduced = ops.max(x, axis=0)
    bias_view = ops.transpose(bias.reshape([4, 1]), 0, 1)
    reshape_into_epilogue.output(reduced + bias_view)


def test_reshape_and_view_fuse_into_opaque_epilogue(
    models: dict[str, Model],
) -> None:
    """The same rule holds for an opaque's epilogue load, not just its prologue.

    Mirrors `sliceReshapeTransposeIntoEpilogue` in
    GraphCompiler/test/mo-opt/MAPDialect/Transforms/ViewFusion/
    view_fusion_reshape_barrier.mlir. `reduce.max` is the opaque; its store-path
    add reads a bias fed by `transpose(reshape(bias))`. `findExistingCapture`
    resolves an epilogue capture the same way it does a prologue one, so the
    reshape and the transpose below it fuse into the epilogue's load path -- and
    a view feeding the reshape would stay materialized just as in the prologue
    case. An e2e correctness check (compiles, matches numpy); the fusion
    structure is covered by the lit test.
    """
    x_np = np.random.randn(4, 4).astype(np.float32)
    bias_np = np.random.randn(4).astype(np.float32)
    (out,) = execute(models[reshape_into_epilogue.name], x_np, bias_np)
    ref = np.max(x_np, axis=0, keepdims=True) + bias_np.reshape([4, 1]).T
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


with _graph(
    "view_and_reshape_into_add", [10, 64], [96]
) as view_and_reshape_into_add:
    a, b = (v.tensor for v in view_and_reshape_into_add.inputs)
    view_and_reshape_into_add.output(a[1:4, 32:64] + b.reshape([3, 32]))


def test_view_and_reshape_both_fuse_into_add(models: dict[str, Model]) -> None:
    """`slice(a) + reshape(b)` fuses both views into the one add kernel.

    The reshape indexes one operand's load, the slice the other -- the reshape
    on one load must not block the slice fusing into the other (the barrier is
    per-load, not per-body). Mirrors `viewAndReshapeFuseIntoAdd` in
    view_fusion_reshape_barrier.mlir.
    """
    model = models[view_and_reshape_into_add.name]
    check_fusion(model, r"mo\.slice.*mo\.add")
    a_np = np.random.randn(10, 64).astype(np.float32)
    b_np = np.random.randn(96).astype(np.float32)
    (out,) = execute(model, a_np, b_np)
    ref = a_np[1:4, 32:64] + b_np.reshape([3, 32])
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


with _graph("stacked_reshapes", [10, 64]) as stacked_reshapes:
    (x,) = (v.tensor for v in stacked_reshapes.inputs)
    y = ops.transpose(x[1:4, 32:64].reshape([16, 6]), 0, 1)
    stacked_reshapes.output(ops.relu(y.reshape([96])))


def test_stacked_reshapes_split_at_each_reshape(
    models: dict[str, Model],
) -> None:
    """`slice -> reshape -> transpose -> reshape -> relu` compiles and is
    correct: each reshape is terminal for its own upstream, so the chain splits
    into groups at each reshape's input rather than fusing across a rank change
    (which `LoopInvariantViewMotion` could not hoist). An e2e correctness check.
    """
    a = np.random.randn(10, 64).astype(np.float32)
    (out,) = execute(models[stacked_reshapes.name], a)
    ref = np.maximum(a[1:4, 32:64].reshape([16, 6]).T.reshape([96]), 0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


with _graph("reshape_producer", [4, 6]) as reshape_producer:
    (x,) = (v.tensor for v in reshape_producer.inputs)
    reshape_producer.output(ops.relu(x.reshape([24])))


def test_reshape_producer_fuses_downstream(models: dict[str, Model]) -> None:
    """A reshape feeding an elementwise consumer fuses in as the producer.

    Reshape is always the terminal (most-upstream) link in a fusion chain --
    nothing fuses into a reshape kernel -- but it may itself fuse into
    whatever consumes its result.
    """
    # No fusion check: a reshape is a pure index transform, so a fused
    # reshape leaves no leaf of its own in the summary -- there is nothing
    # distinguishing this from an unfused reshape to match on.
    a = np.random.randn(4, 6).astype(np.float32)
    (out,) = execute(models[reshape_producer.name], a)
    np.testing.assert_allclose(
        out, np.maximum(a.reshape([24]), 0), rtol=1e-5, atol=1e-5
    )
