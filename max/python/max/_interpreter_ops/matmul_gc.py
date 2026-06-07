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

"""Graph-compiler matmul model cache for the MO interpreter.

At import time a single ``load_all`` compiles every supported (device, dtype)
batched-matmul graph once, amortizing compiler cold-start. The compiled models
are cached and served to the eager ``mo.matmul`` / ``mo.batch_matmul`` handler
via :func:`matmul_model`. This module must not import from ``handlers.py``.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from max import engine
from max._mlir_context import in_default_mlir_context
from max.driver import Device, DeviceSpec, accelerator_count, load_devices
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "matmul"

_ACCELERATOR_DEVICES = load_devices(
    [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
)

_ACCELERATOR_DTYPES = [DType.float32, DType.float16, DType.bfloat16]

_CPU_DEVICES = load_devices([DeviceSpec.cpu()])

# Conservative set proven to compile on every CI architecture. float16/bfloat16
# fail matmul kernel codegen on ARM, so widen only with per-arch CI confirmation.
_CPU_DTYPES = [
    DType.float32,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
]


@dataclass(frozen=True)
class CompilationTarget:
    graph_op_name: str
    device: Device
    # A single dtype shared by both operands. In principle lhs and rhs can
    # have different dtypes. In that case extend the dataclass
    dtype: DType

    @property
    def graph_name(self) -> str:
        """Returns the string used both as the graph ``sym_name`` and cache key."""
        return f"{self.graph_op_name}_{self.device.label}_{self.device.id}_{self.dtype}"


_COMPILATION_TARGETS = [
    CompilationTarget(_GRAPH_BASE_NAME, device, dtype)
    for device, dtype in itertools.chain(
        itertools.product(_CPU_DEVICES, _CPU_DTYPES),
        itertools.product(_ACCELERATOR_DEVICES, _ACCELERATOR_DTYPES),
    )
]

_MATMUL_MODEL_CACHE: dict[str, engine.Model] = {}


def canonical_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    """Flattens an arbitrary-rank matmul operand to canonical rank 3.

    ``[d0, ..., dn, i, j]`` becomes ``(d0*...*dn, i, j)``; a rank-2 ``[i, j]``
    becomes ``(1, i, j)`` because ``prod(())`` is the empty product ``1``,
    keeping the rank-2 case branchless.
    """
    *batch_dims, i, j = shape
    return (prod(batch_dims), i, j)


def _build_matmul_graph(
    module: Module, compilation_target: CompilationTarget
) -> None:
    """Adds one fully-symbolic rank-3 matmul graph into *module* in-place."""
    dev_ref = DeviceRef.from_device(compilation_target.device)
    lhs_type = TensorType(
        compilation_target.dtype, ["batch", "m", "k"], device=dev_ref
    )
    rhs_type = TensorType(
        compilation_target.dtype, ["batch", "k", "n"], device=dev_ref
    )
    graph_name = compilation_target.graph_name
    g = Graph(graph_name, input_types=[lhs_type, rhs_type], module=module)
    with g:
        lhs, rhs = g.inputs
        g.output(graph_ops.matmul(lhs.tensor, rhs.tensor))


@in_default_mlir_context
def compile_matmul_sweep() -> None:
    """Compile every supported (device, dtype) matmul combination in one shot.

    Builds CPU plus every accelerator, adds one fully-symbolic rank-3
    batched-matmul graph per supported (device spec, dtype) target into a
    single :class:`~max.graph.Module`, then calls
    ``InferenceSession.load_all`` once. All compiled
    :class:`~max.engine.Model` objects land in ``_MATMUL_MODEL_CACHE`` keyed
    by :attr:`CompilationTarget.graph_name`.

    The sweep runs once at import time and populates the cache for the
    lifetime of the process.  There is no lazy compile path; a cache miss
    at dispatch time is a hard error.
    """
    module = Module()
    for compilation_target in _COMPILATION_TARGETS:
        _build_matmul_graph(module, compilation_target)

    devices = {
        compilation_target.device for compilation_target in _COMPILATION_TARGETS
    }
    session = engine.InferenceSession(devices=devices)
    _MATMUL_MODEL_CACHE.update(session.load_all(module, weights_registry={}))


def matmul_model(device: Device, dtype: DType) -> engine.Model:
    """Return the pre-compiled matmul :class:`~max.engine.Model` for *device* + *dtype*.

    The model is retrieved from ``_MATMUL_MODEL_CACHE``, which is populated
    once at module import time by :func:`compile_matmul_sweep`.

    Args:
        device: The target device (CPU or GPU accelerator).
        dtype: The element dtype for both operands.

    Returns:
        The compiled :class:`~max.engine.Model` ready for execution.

    Raises:
        KeyError: If *device* / *dtype* was not included in the sweep (e.g.
            ``float64`` on GPU, or an NPU device).  The error message names
            the exact key and lists what *is* available.
    """
    key = CompilationTarget(_GRAPH_BASE_NAME, device, dtype).graph_name
    model = _MATMUL_MODEL_CACHE.get(key)
    if model is None:
        raise KeyError(
            f"No pre-compiled matmul model for key {key!r}."
            f"  Available: {sorted(_MATMUL_MODEL_CACHE)}"
        )
    return model
