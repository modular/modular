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

"""Graph-compiler rms_norm model cache for the MO interpreter.

Covers ``ReduceRmsNormOp`` only. ``epsilon``, ``weight_offset`` and
``multiply_before_cast`` are compile-time attributes of the MO op, so all
three ride in the cache key. ``epsilon`` and ``weight_offset`` are keyed by
their ``f32`` bit patterns, which is exactly what the op stores.

The precompile sweep covers the scalar values models use in practice (see
``_SWEPT_SCALARS``); any other value compiles lazily on first use.

Supported dtypes mirror the shared float set (see
:func:`gc_compile.float_dtypes`), same policy as ``layer_norm_gc``.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

import struct

from max import engine
from max._core.dialects import builtin, kgen, mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph.ops.rms_norm import _float32_attr

canonical_rank2 = gc_compile.canonical_rank2

_VARIANTS = (False, True)  # multiply_before_cast: Llama-style, Gemma-style

# (epsilon, weight_offset) pairs the precompile sweep covers.
_SWEPT_SCALARS = (
    (1e-5, 0.0),
    (1e-6, 0.0),
    (1e-5, 1.0),
    (1e-6, 1.0),
)


def _f32_bits(value: float) -> str:
    return struct.pack("<f", value).hex()


def _graph_name(
    device: Device,
    dtype: DType,
    epsilon: float,
    weight_offset: float,
    multiply_before_cast: bool,
) -> str:
    return (
        f"rms_norm_{device.label}_{device.id}_{dtype.name}"
        f"_eps{_f32_bits(epsilon)}_wo{_f32_bits(weight_offset)}"
        f"_mbc{int(multiply_before_cast)}"
    )


def _rms_norm_graph(
    module: Module,
    device: Device,
    dtype: DType,
    epsilon: float,
    weight_offset: float,
    multiply_before_cast: bool,
) -> None:
    """Adds one fully-symbolic rank-2 rms_norm graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    graph = Graph(
        _graph_name(
            device, dtype, epsilon, weight_offset, multiply_before_cast
        ),
        input_types=[
            TensorType(dtype, ["rows", "features"], device=device_ref),
            TensorType(dtype, ["features"], device=device_ref),
        ],
        module=module,
    )
    with graph:
        x, weight = (v.tensor for v in graph.inputs)
        result = Graph.current._add_op_generated(
            mo.ReduceRmsNormOp,
            result=TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            input=x,
            weight=weight,
            epsilon=_float32_attr(epsilon),
            weight_offset=_float32_attr(weight_offset),
            multiply_before_cast=builtin.BoolAttr(multiply_before_cast),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


class _RmsNormFamily(gc_compile.GCFamilySpec):
    name = "rms_norm"

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
        for dtype in gc_compile.float_dtypes(device):
            for epsilon, weight_offset in _SWEPT_SCALARS:
                for multiply_before_cast in _VARIANTS:
                    _rms_norm_graph(
                        module,
                        device,
                        dtype,
                        epsilon,
                        weight_offset,
                        multiply_before_cast,
                    )
        return module


_FAMILY = gc_compile.GCOpFamily(_RmsNormFamily())
gc_compile.register_family(_FAMILY)


def rms_norm_model(
    device: Device,
    dtype: DType,
    epsilon: float,
    weight_offset: float,
    multiply_before_cast: bool,
) -> engine.Model:
    """Returns the rms_norm Model for the given target.

    Lazy by default (compiled and cached on first use); the first miss adopts
    a whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: The realized input's device.
        dtype: The realized input's dtype.
        epsilon: The op's compile-time ``epsilon``.
        weight_offset: The op's compile-time ``weight_offset``.
        multiply_before_cast: The op's compile-time ``multiply_before_cast``
            flag (Gemma-style vs Llama-style).

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If (device, dtype) is outside the supported set; or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target was not
            swept.
    """
    key = _graph_name(
        device, dtype, epsilon, weight_offset, multiply_before_cast
    )
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        supported = gc_compile.float_dtypes(device)
        if dtype in supported:
            return None
        return (
            f"Unsupported rms_norm device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {supported}"
        )

    def build(module: Module) -> None:
        _rms_norm_graph(
            module, device, dtype, epsilon, weight_offset, multiply_before_cast
        )

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
