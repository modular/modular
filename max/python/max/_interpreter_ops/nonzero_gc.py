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

"""Graph-compiler arg_nonzero model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

The model keeps the input rank so the kernel writes final ``[nnz, rank]``
coordinates directly on the input device. Every dimension is symbolic, so one
model serves all shapes of a given rank. Ranks above
:data:`gc_compile.MAX_RANK` are never precompiled: they compile lazily on
demand, and are rejected with a clear error only when this process disallows
lazy compile (warm/precompile mode).

The output row count is data-dependent: the op's registered shape function
returns the exact ``nnz`` (``builtin_kernels/reductions.mojo``), so the model
sizes its own ``[nnz, rank]`` output.

Dtype axis: floats keep their real dtype, never bitcast -- ``-0.0`` is zero as
a float but a nonzero bit pattern as an unsigned int, so a bitcast would
misclassify it. Float64 is swept on non-Metal accelerators too: nonzero only
loads and compares against zero, so the f64 exclusion in
:func:`gc_compile.float_dtypes` (MSTDL-2711, about other ops; Metal simply
lacks f64) doesn't apply, and eager GPU f64 worked before this family (the
handler used to round-trip inputs through the host). Signed ints, unsigned
ints, and bool each have a unique zero bit pattern, so they ride the four
uint widths instead
(:func:`gc_compile.uint_view_dtype` / :data:`gc_compile.WIDTH_DTYPES`), exactly
as ``band_part_gc`` does for its bit-width-keyed masking. The sweep is keyed
by (device, dtype, rank): per device, :func:`_supported_dtypes` (the real
float dtypes plus the 4 uint widths), each at ranks
1..:data:`gc_compile.MAX_RANK`.
"""

from dataclasses import dataclass

from max import _eager_policy, engine
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "nonzero"


def _supported_dtypes(device: Device) -> list[DType]:
    """Returns the real float dtypes and integer view widths for ``device``.

    Floats: ``gc_compile.float_dtypes(device)`` plus f64 on non-Metal
    accelerators -- nonzero only loads and compares against zero, so the f64
    exclusion behind ``gc_compile.GPU_FLOAT_DTYPES`` (MSTDL-2711, about other
    ops) doesn't apply here, and eager GPU f64 must keep working. Metal still
    lacks f64. Signed ints, unsigned ints, and bool ride the uint view
    widths regardless.
    """
    floats = gc_compile.float_dtypes(device)
    if device.label != "cpu" and device.api != "metal":
        floats = [*floats, DType.float64]
    return floats + gc_compile.WIDTH_DTYPES


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType
    """The real dtype for floats, or the uint width it's viewed as for
    int/bool (see :func:`gc_compile.uint_view_dtype`)."""
    rank: int
    """Input rank; every dimension is symbolic, so one graph serves all
    shapes of this rank."""

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype.name}_r{self.rank}"
        )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype, rank)
    for device in gc_compile.DISCOVERED_DEVICES
    for dtype in _supported_dtypes(device)
    for rank in range(1, gc_compile.MAX_RANK + 1)
]


def _nonzero_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one rank-polymorphic-shape arg_nonzero graph to ``module``."""
    graph = Graph(
        target.graph_name,
        input_types=[
            TensorType(
                target.dtype,
                [f"d{i}" for i in range(target.rank)],
                device=DeviceRef.from_device(target.device),
            )
        ],
        module=module,
    )
    with graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(graph_ops.nonzero(x, "nnz"))


class _NonzeroFamily(gc_compile.GCFamilySpec):
    name = "nonzero"

    def sweep_devices(self) -> list[Device]:
        return list(gc_compile.DISCOVERED_DEVICES)

    def build_module(self) -> Module:
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        if module is None:
            module = Module()
        for target in _COMPILATION_TARGETS:
            if (
                target.device.label == device.label
                and target.device.id == device.id
            ):
                _nonzero_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_NonzeroFamily())
gc_compile.register_family(_FAMILY)


def nonzero_model(device: Device, dtype: DType, rank: int) -> engine.Model:
    """Returns the arg_nonzero model for ``device``, ``dtype``, and ``rank``.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: Device on which the input and output reside.
        dtype: The input element dtype: a real float dtype, or the uint width
            an int/bool input is bit-cast to (see :func:`gc_compile.
            uint_view_dtype`).
        rank: Input rank. Ranks above ``gc_compile.MAX_RANK`` are not
            precompiled; they compile on demand, except when this process
            disallows lazy compile (``MAX_EAGER_ALLOW_LAZY_COMPILE=0``).

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If *dtype* is outside the supported set; or if *rank*
            exceeds ``gc_compile.MAX_RANK`` while this process disallows
            lazy compile (``MAX_EAGER_ALLOW_LAZY_COMPILE=0``); or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not
            precompiled.
    """
    target = CompilationTarget(device, dtype, rank)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if rank < 1:
            return f"arg_nonzero requires rank >= 1; got rank {rank}."
        if (
            rank > gc_compile.MAX_RANK
            and not _eager_policy.allow_lazy_compile()
        ):
            return (
                f"arg_nonzero precompiles rank <= {gc_compile.MAX_RANK};"
                f" got rank {rank}. Higher ranks compile on demand only,"
                " which this process disallows"
                f" ({_eager_policy.ALLOW_LAZY_COMPILE_ENV_VAR}=0)."
            )
        if dtype not in _supported_dtypes(device):
            supported = ", ".join(str(d) for d in _supported_dtypes(device))
            return (
                f"arg_nonzero does not support dtype {dtype}; supported:"
                f" {supported}. (key {key!r})"
            )
        return None

    return _FAMILY.model_for(
        key,
        device,
        lambda m: _nonzero_graph(m, target),
        unsupported_reason=check_supported,
    )
