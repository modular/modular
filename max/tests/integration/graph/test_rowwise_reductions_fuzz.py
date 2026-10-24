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
"""Property tests for the sum, max, min and mean reductions.

These ops compile to rowwise kernels that the graph compiler generates, so
they're fuzzed through the graph. Each case compiles one symbolic-dimension
graph, then hypothesis draws the extents and the values. Extents start at 0:
a reduction over an empty axis still owns ``product(shape \\ axis)`` outputs,
and each must be written with the reduction's identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from hypothesis import example, given, settings
from hypothesis import strategies as st
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType
from test_common.reduction_graphs import build_reduction

_OPS = ("reduce_sum", "reduce_max", "reduce_min", "reduce_mean")


@dataclass(frozen=True)
class FuzzSpec:
    """One compiled graph: an op over one axis of a rank-`rank` input."""

    op: str
    rank: int
    axis: int
    device: str

    @property
    def name(self) -> str:
        return f"{self.op}_rank{self.rank}_axis{self.axis}_{self.device}"


# The inner axis, a non-inner one, and the middle of a rank-3 input, plus a
# whole rank-1 tensor, which is what `ops.sum` of a flat tensor lowers to.
_SHAPES = [(2, 1), (2, 0), (3, 1)]
SPECS = [
    FuzzSpec(op, rank, axis, device)
    for device in ("cpu", "gpu")
    for op in _OPS
    for rank, axis in _SHAPES
] + [FuzzSpec("reduce_sum", 1, 0, device) for device in ("cpu", "gpu")]

# Row lengths around the SIMD, warp and block boundaries the GPU tiers pivot
# on, so drawn shapes hit them more often than uniform draws would.
_AXIS_EXTENTS = st.one_of(
    st.sampled_from([0, 1, 31, 32, 33, 127, 128, 129, 1023, 1024, 1025]),
    st.integers(min_value=0, max_value=4096),
)
_OTHER_EXTENTS = st.integers(min_value=0, max_value=16)
_DISTS = ("normal", "uniform", "sparse", "large", "constant")


def _build_graph(spec: FuzzSpec) -> Graph:
    device = DeviceRef.CPU() if spec.device == "cpu" else DeviceRef.GPU()
    input_type = TensorType(
        DType.float32, [f"d{i}" for i in range(spec.rank)], device=device
    )
    with Graph(spec.name, input_types=[input_type]) as graph:
        x = graph.inputs[0].tensor
        graph.output(build_reduction(spec.op, x, spec.axis))
    return graph


def _make_input(shape: tuple[int, ...], dist: str, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if dist == "uniform":
        return torch.rand(shape, generator=gen) * 2 - 1
    if dist == "sparse":
        x = torch.randn(shape, generator=gen)
        return x * (torch.rand(shape, generator=gen) < 0.05)
    if dist == "large":
        return torch.randn(shape, generator=gen) * 1e4
    if dist == "constant":
        return torch.full(shape, 0.75)
    return torch.randn(shape, generator=gen)


def _run(model: Model, x: torch.Tensor) -> np.ndarray:
    out = model.execute(Buffer.from_dlpack(x).to(model.input_devices[0]))[0]
    assert isinstance(out, Buffer)
    return (out if out.device.is_host else out.to(CPU())).to_numpy()


def _check(op: str, x: torch.Tensor, axis: int, got: np.ndarray) -> None:
    """Compares `got` to a float64 reference, with identities on an empty axis.

    A sum's error is bounded by `n * eps * sum(|x|)` whatever the order of the
    additions, so that bound never fails a correct kernel; a dropped or doubled
    element is far larger. Max and min are exact.
    """
    x64 = x.to(torch.float64)
    n = x.shape[axis]
    eps = float(np.finfo(np.float32).eps)
    abs_sum = x64.abs().sum(dim=axis, keepdim=True).numpy()
    if op == "reduce_sum":
        ref = x64.sum(dim=axis, keepdim=True).numpy()
        tol = n * eps * abs_sum + eps * np.abs(ref)
    elif op == "reduce_mean":
        if n == 0:
            assert np.isnan(got).all(), "mean over an empty axis must be NaN"
            return
        ref = x64.mean(dim=axis, keepdim=True).numpy()
        tol = eps * abs_sum + 2 * eps * np.abs(ref)
    elif n == 0:
        identity = np.finfo(np.float32)
        ref = np.full(
            got.shape, identity.min if op == "reduce_max" else identity.max
        )
        tol = np.zeros(got.shape)
    else:
        reduce = torch.amax if op == "reduce_max" else torch.amin
        ref = reduce(x64, dim=axis, keepdim=True).numpy()
        tol = np.zeros(ref.shape)
    assert got.shape == ref.shape, f"shape {got.shape} != {ref.shape}"
    bad = np.abs(got.astype(np.float64) - ref) > tol
    assert not bad.any(), (
        f"{int(bad.sum())}/{bad.size} outputs off; first at "
        f"{np.argwhere(bad)[0].tolist()}: got {got[bad][0]}, want {ref[bad][0]}"
    )


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_rowwise_reductions_fuzz(
    session: InferenceSession, spec: FuzzSpec
) -> None:
    model = session.load(_build_graph(spec))
    # One run outside hypothesis: when `precompile_mefs` records this test
    # without hardware, the first execute stops it, and hypothesis would
    # report that stop as a failure instead.
    _run(model, torch.zeros((1,) * spec.rank))

    # Same examples on every run, so a failure is never another PR's flake.
    @settings(max_examples=40, derandomize=True, deadline=None)
    @given(
        axis_extent=_AXIS_EXTENTS,
        other_extents=st.lists(_OTHER_EXTENTS, min_size=2, max_size=2),
        dist=st.sampled_from(_DISTS),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    # Shapes that once exposed `rowwise.launch` bugs: an empty axis returned
    # before writing any output, and 33 dropped a row to the scalar path. On
    # a non-inner axis, 3 and 4096 columns reach the GPU block and tiled tiers.
    @example(axis_extent=0, other_extents=[16, 16], dist="normal", seed=0)
    @example(axis_extent=0, other_extents=[3, 3], dist="normal", seed=0)
    @example(axis_extent=0, other_extents=[4096, 1], dist="normal", seed=0)
    @example(axis_extent=33, other_extents=[8, 8], dist="normal", seed=0)
    def check(
        axis_extent: int, other_extents: list[int], dist: str, seed: int
    ) -> None:
        others = iter(other_extents)
        shape = tuple(
            axis_extent if dim == spec.axis else next(others)
            for dim in range(spec.rank)
        )
        x = _make_input(shape, dist, seed)
        if axis_extent == 0 and all(
            extent for dim, extent in enumerate(shape) if dim != spec.axis
        ):
            # A GPU output recycled from an earlier launch keeps that launch's
            # values when an empty reduction skips its writes.
            warm = list(shape)
            warm[spec.axis] = 3
            _run(model, _make_input(tuple(warm), "large", seed))
        _check(spec.op, x, spec.axis, _run(model, x))

    check()
