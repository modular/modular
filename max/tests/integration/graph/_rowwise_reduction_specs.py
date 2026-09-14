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
"""Shared spec list and graph construction for the row-wise reduction tests.

The CPU producer (``precompile_rowwise_reductions``) and the GPU consumer
(``test_rowwise_reductions``) both import this module so they build the
*identical* graph -- the producer compiles each spec to a MEF with no GPU
attached, the consumer initializes that MEF and executes it. Keeping the spec
list and the construction in one place is what guarantees the compiled
artifact and the runtime inputs can't drift apart.

Every graph here is weightless: the ``layer_norm`` and ``rms_norm`` parameters
are graph *inputs*, not weights, so the consumer initializes with no weights
registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from test_common.reduction_graphs import PURE_REDUCTIONS, build_reduction


@dataclass(frozen=True)
class RowwiseSpec:
    """One compiled-graph parametrization of a row-wise reduction."""

    name: str
    """Stable identifier used for the MEF filename and the test id."""

    op: str
    """Reduction op name, as understood by ``build_reduction``."""

    dtype: DType
    """Element type of the graph input."""

    axis: int
    """Reduction axis: ``-1`` for the inner axis, ``0`` for a non-inner one."""

    device: DeviceRef
    """Device the graph is pinned to."""


_DTYPE_TAGS = {
    DType.bfloat16: "bf16",
    DType.float16: "f16",
    DType.float32: "f32",
    DType.float64: "f64",
    DType.int64: "i64",
    DType.bool: "bool",
}
_AXIS_TAGS = {-1: "inner", 0: "noninner"}
_DEVICE_REFS = {"cpu": DeviceRef.CPU(), "gpu": DeviceRef.GPU()}


def _spec(op: str, dtype: DType, axis: int, device: str) -> RowwiseSpec:
    return RowwiseSpec(
        f"{op}_{_DTYPE_TAGS[dtype]}_{_AXIS_TAGS[axis]}_{device}",
        op,
        dtype,
        axis,
        _DEVICE_REFS[device],
    )


# GPU is the priority target: every cooperative tier (warp / block), the
# cross-thread combine and the split-K path exist only there, while the CPU path
# is a serial per-row walk. So the inner and non-inner matrices below are built
# for both devices.
_BOTH_DEVICES = ("cpu", "gpu")

# Inner-axis matrix: precision-sensitive ops in both dtypes; exact
# selection/compare ops (max/min_and_max/arg) in bfloat16 only.
#
# reduce_min/reduce_mean/logsoftmax/argmin are deliberately absent: each is
# redundant with a sibling already in this matrix (reduce_max, reduce_sum,
# softmax, argmax respectively) -- same monoid/combine mechanics, differing
# only in comparison direction or a trivial final step (a divide, or the
# elementwise map after the shared OnlineLogSumExp reduce). A bug in the
# shared scaffolder path shows up in the sibling that's still tested; the two
# confirmed bugs that genuinely needed both directions tested (the bool
# subword-state crash, the split-K join_parallel bug) have their own narrow
# regression coverage below (`_GPU_SUBWORD_CASES`, `_SPLITK_OPS`) that still
# exercises both regardless of this trim.
_INNER_BOTH = [
    "reduce_sum",
    "reduce_product",
    "softmax",
    "layer_norm",
    "rms_norm",
    "row_mean_of_squares",
]
_INNER_BF16_ONLY = [
    "reduce_max",
    "argmax",
    "reduce_min_and_max",
]
_INNER_CASES = (
    [(op, DType.bfloat16) for op in _INNER_BOTH + _INNER_BF16_ONLY]
    + [(op, DType.float32) for op in _INNER_BOTH]
    + [(op, DType.float16) for op in _INNER_BOTH]
)

INNER_SPECS = [
    _spec(op, dtype, -1, device)
    for device in _BOTH_DEVICES
    for op, dtype in _INNER_CASES
]

# Non-inner (reduce over axis 0): the pure arbitrary-axis reductions. Precision-
# sensitive sum in both dtypes; the rest in bfloat16 only. See the inner-axis
# matrix above for why reduce_min/reduce_mean/argmin are absent (redundant
# with reduce_max/reduce_sum/argmax).
_NONINNER_CASES = [
    (op, DType.bfloat16)
    for op in [
        "reduce_sum",
        "reduce_max",
        "reduce_product",
        "argmax",
        "reduce_min_and_max",
    ]
] + [(op, DType.float32) for op in ["reduce_sum"]]

NONINNER_SPECS = [
    _spec(op, dtype, 0, device)
    for device in _BOTH_DEVICES
    for op, dtype in _NONINNER_CASES
]

# 8-byte elements, both axes -- CPU only (MAX's reduce dtype set has no 64-bit
# element on GPU). The widest element is what turns a wrong tile alignment into a
# crash rather than a slow path: the alignment a body claims per tile is scaled
# by the element size before it reaches the backend, so only on the widest
# element can an off-by-a-factor claim reach a whole SIMD register and select an
# alignment-checked instruction. Nothing narrower can catch that, which is why
# every dtype above passed while float64 segfaulted. Both axes, since the claim
# rides every load, and both 8-byte dtypes, since the claim is a function of the
# element's width and not of its kind -- int64 faulted on the same ops.
#
# int64 skips reduce_mean and reduce_product: integer division and integer
# overflow make them a comparison against the float32 reference rather than
# against the alignment claim, which every other op here already pins.
_WIDE_CASES = [(op, DType.float64) for op in PURE_REDUCTIONS] + [
    (op, DType.int64)
    for op in PURE_REDUCTIONS
    if op not in ("reduce_mean", "reduce_product")
]

WIDE_CPU_SPECS = [
    _spec(op, dtype, axis, "cpu")
    for axis in (-1, 0)
    for op, dtype in _WIDE_CASES
]

# Sub-word monoid state, on an accelerator. The Row scaffolder's cross-thread
# combine (`Reducer.generic`, max/kernels/src/algorithm/gpu/rowwise.mojo) exists
# only on GPU and exchanges the monoid's width-1 state in whole 4-byte words, so
# a monoid whose state is narrower than one word is reachable only here -- the
# CPU cases above cannot see it, and it is silent memory corruption rather than a
# crash. Only the inner axis reaches the combine: a non-inner reduction lands on
# the tiled tier, where `rowwise.pjoin` no-ops because each thread owns its own
# output column.
#
# Which (op, dtype) pairs have a sub-word state, per each monoid's
# `join_parallel` in max/kernels/src/algorithm/reduce_op.mojo:
#   reduce_product  -- no hardware-fast scalar product exists, so every dtype
#                      takes the combine; fp16/bf16 make the state 2 bytes.
#   reduce_max/min  -- the fast-reducer dtype list covers only the 32/64-bit
#                      dtypes, so `bool` (1 byte) falls through.
#
# The sub-word integers are the same bug class -- `int8`/`int16` are legal graph
# op inputs for `reduce_max`/`reduce_min`/`reduce_min_and_max` and sit outside
# every fast-reducer dtype list -- but the legacy kernels that still handle
# these ops cannot compile them, so those cases arrive with the change that
# routes each op through the Row API (`bool` is not legal for `min_and_max`).
_GPU_SUBWORD_CASES = [
    ("reduce_product", DType.float16),
    ("reduce_product", DType.bfloat16),
    ("reduce_max", DType.bool),
    ("reduce_min", DType.bool),
]

GPU_SUBWORD_SPECS = [
    _spec(op, dtype, -1, "gpu") for op, dtype in _GPU_SUBWORD_CASES
]

# Split-K cross-block join, on an accelerator. argmax/argmin are what catch a
# mistake there: their cross-thread step ends by publishing the winning index
# into the field the emit reads, so a finish that skips it returns a
# real-but-not-extreme index -- a wrong answer that still looks like a plausible
# one. reduce_max rides the same call with an exactly representable result, so
# it pins the non-arg monoids on that path too.
_SPLITK_OPS = ["argmax", "argmin", "reduce_max"]

GPU_SPLITK_SPECS = [_spec(op, DType.float32, -1, "gpu") for op in _SPLITK_OPS]

SPECS_BY_NAME: dict[str, RowwiseSpec] = {
    spec.name: spec
    for spec in (
        *INNER_SPECS,
        *NONINNER_SPECS,
        *WIDE_CPU_SPECS,
        *GPU_SUBWORD_SPECS,
        *GPU_SPLITK_SPECS,
    )
}


def build_graph(spec: RowwiseSpec) -> Graph:
    """Builds ``spec``'s single-reduction graph with symbolic dimensions.

    This is the single construction path the CPU producer compiles and the GPU
    consumer initializes.

    Args:
        spec: The parametrization to build.

    Returns:
        The constructed :class:`~max.graph.Graph`, ready to compile.
    """
    x_t = TensorType(spec.dtype, ["r", "c"], device=spec.device)
    w_t = TensorType(spec.dtype, ["c"], device=spec.device)
    if spec.op == "layer_norm":
        input_types = [x_t, w_t, w_t]  # x, gamma, beta
    elif spec.op == "rms_norm":
        input_types = [x_t, w_t]
    else:
        input_types = [x_t]

    with Graph(spec.name, input_types=input_types) as graph:
        x = graph.inputs[0].tensor
        weights = [inp.tensor for inp in graph.inputs[1:]]
        graph.output(build_reduction(spec.op, x, spec.axis, weights=weights))
    return graph
