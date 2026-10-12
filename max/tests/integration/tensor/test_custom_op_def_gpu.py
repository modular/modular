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
"""Accelerator tests for `max.experimental.custom.declare` bindings.

The sibling CPU suite (``test_custom_op_def.py``) forces
``CUDA_VISIBLE_DEVICES=""``, so device-conditional coverage (the binding
key includes ``device``) has to live here instead. Beyond
``op_with_external_cubin``, this suite runs the three ``foreach``-based
NVFP4 pipeline kernels (quantize, matmul, relu) on the accelerator, since
those are device-portable.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from max import _core, engine
from max._interpreter_ops import custom_gc
from max._interpreter_ops import handlers as _handlers
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.experimental import custom as C
from max.experimental import executor
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.experimental.testing import assert_all_close
from max.graph import DeviceRef, Graph, TensorType, TensorValue

KERNEL_VERIFICATION_OPS = Path(
    os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"]
)


_CustomOpResult = Tensor | list[Tensor] | TensorValue | list[TensorValue]


def _one_tensor(result: _CustomOpResult) -> Tensor:
    """Narrows an eager `CustomOp` call's single-output result to `Tensor`."""
    assert isinstance(result, Tensor)
    return result


def _one_value(result: _CustomOpResult) -> TensorValue:
    """Narrows a graph-staged single-output result to `TensorValue`."""
    assert isinstance(result, TensorValue)
    return result


def _two_tensors(result: _CustomOpResult) -> tuple[Tensor, Tensor]:
    """Narrows an eager two-output result to a pair of `Tensor`s."""
    assert isinstance(result, list) and len(result) == 2
    first, second = result
    assert isinstance(first, Tensor) and isinstance(second, Tensor)
    return first, second


def _two_values(result: _CustomOpResult) -> tuple[TensorValue, TensorValue]:
    """Narrows a graph-staged two-output result to a pair of `TensorValue`s."""
    assert isinstance(result, list) and len(result) == 2
    first, second = result
    assert isinstance(first, TensorValue) and isinstance(second, TensorValue)
    return first, second


@pytest.fixture(autouse=True)
def _clear_binding_cache() -> Iterator[None]:
    """Each test gets a clean cache: ``_CACHE`` is process-global, so a hit
    left behind by an earlier test would hide a real miss (mirrors the CPU
    suite's fixture)."""
    custom_gc._CACHE.clear()
    yield
    custom_gc._CACHE.clear()


@pytest.fixture
def force_interpreter_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forces every eager call through the interpreter, with no compile
    fallback.

    ``MAX_EAGER_EXECUTOR=interpreter`` has no effect on already-running
    Python, since ``default_executor()`` reads a module global set once at
    import time. Patching it directly is the only way to prove the
    interpreter served the call.
    """
    monkeypatch.setattr(
        executor,
        "_DEFAULT_EXECUTOR",
        executor.InterpreterExecutor(max_ops=None),
    )


def make_vec_add() -> C.CustomOp:
    """Shape- and dtype-preserving rank-1 vector add, run on the device."""
    (n,) = C.Symbols("n")
    return C.declare(
        "op_with_external_cubin",
        inputs={
            "lhs": C.TemplateType(DType.float32, [n]),
            "rhs": C.TemplateType(DType.float32, [n]),
        },
        outputs=[C.TemplateType(DType.float32, [n])],
        custom_extensions=[KERNEL_VERIFICATION_OPS],
    )


def _spy_on_compiles(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts real ``custom_gc._compile`` calls.

    ``len(_CACHE)`` alone can't catch a binding that recompiles onto the
    same key every call: that would overwrite (not grow) the dict, so the
    size would still read back as expected.
    """
    counter = [0]
    original = custom_gc._compile

    def counting_compile(*args: object, **kwargs: object) -> engine.Model:
        counter[0] += 1
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(custom_gc, "_compile", counting_compile)
    return counter


def test_declared_binding_serves_two_shapes_on_accelerator(
    force_interpreter_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One accelerator binding, two shapes, correct numerics for both: the
    binding is rank-polymorphic, so the second shape reuses the first's
    compiled model, and each result must still be right to rule out a
    stale binding returning the previous shape's output.
    """
    device = Accelerator()
    op = make_vec_add()
    compiles = _spy_on_compiles(monkeypatch)

    zeros_64 = Tensor.zeros([64], dtype=DType.float32, device=device)
    ones_64 = Tensor.ones([64], dtype=DType.float32, device=device)
    out_64 = _one_tensor(op(zeros_64, ones_64))
    assert compiles[0] == 1
    assert [int(d) for d in out_64.shape] == [64]
    assert_all_close(out_64, ones_64)

    twos_128 = Tensor.full([128], 2.0, dtype=DType.float32, device=device)
    threes_128 = Tensor.full([128], 3.0, dtype=DType.float32, device=device)
    out_128 = _one_tensor(op(twos_128, threes_128))
    assert compiles[0] == 1  # same rank/dtype/device: no second compile
    assert len(custom_gc._CACHE) == 1
    assert [int(d) for d in out_128.shape] == [128]
    assert_all_close(
        out_128, Tensor.full([128], 5.0, dtype=DType.float32, device=device)
    )


def test_undeclared_custom_op_compiles_on_accelerator(
    force_interpreter_only: None,
) -> None:
    """The same kernel staged without a ``CustomOp`` has no shape contract
    to build a binding from, so the interpreter refuses it on the device path
    too and leaves ``_CACHE`` untouched; a declared call populates it."""
    device = Accelerator()
    zeros = Tensor.zeros([64], dtype=DType.float32, device=device)
    ones = Tensor.ones([64], dtype=DType.float32, device=device)

    declared = _one_tensor(make_vec_add()(zeros, ones))
    assert_all_close(declared, ones)
    assert len(custom_gc._CACHE) == 1

    with pytest.raises(
        executor.UnsupportedGraphError, match=r"no custom\.declare declaration"
    ):
        F.custom(
            "op_with_external_cubin",
            device=device,
            values=[zeros, ones],
            out_types=[
                TensorType(DType.float32, [64], DeviceRef.from_device(device))
            ],
            custom_extensions=KERNEL_VERIFICATION_OPS,
        )
    assert len(custom_gc._CACHE) == 1


def test_check_realized_shape_device_mismatch_raises() -> None:
    """A realized buffer on a different device than the staged type declared
    must be refused rather than trusted. Moved here from the CPU suite,
    where ``cpu_only = True`` made this case unreachable.

    A hard ``RuntimeError``, like the sibling rank/dtype/shape cases: falling
    back to compilation here would rerun a graph that may already have
    mutated a buffer.
    """
    staged_type = TensorType(DType.float32, [4], DeviceRef.CPU())
    graph = Graph("g_device_probe", input_types=[staged_type])
    with graph:
        graph.output(graph.inputs[0])
    staged: _core.Value[Any] = graph.inputs[0]._mlir_value

    wrong_device = Buffer.zeros((4,), DType.float32, Accelerator())
    with pytest.raises(RuntimeError, match="device mismatch") as excinfo:
        _handlers._check_realized_shape(wrong_device, staged)
    assert not isinstance(excinfo.value, executor.UnsupportedGraphError)


# --- Device mixing and the NVFP4 pipeline -----------------------------------

KERNELS = Path(os.environ["MODULAR_CUSTOM_OP_DEF_KERNELS_PATH"])

_BLOCK = 16
_N = 16  # weight row count for the NVFP4 test; must stay a multiple of _BLOCK
_NVFP4_MAX = 6.0
#: Magnitude gained when an E2M1 code steps from `i` to `i + 1`.
_LEVEL_STEPS = (0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0)
#: Midpoints between neighbouring magnitudes. A tie rounds to the larger one,
#: matching the kernel's strict `<` comparisons.
_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _level(code: Tensor) -> Tensor:
    """Returns the E2M1 magnitude each code stands for.

    Accumulating the steps below a code avoids needing to index a table of
    levels by a runtime value.
    """
    level = Tensor.zeros_like(code)
    for i, step in enumerate(_LEVEL_STEPS):
        level = level + (code > float(i)).cast(code.dtype) * step
    return level


def _code(magnitude: Tensor) -> Tensor:
    """Returns the E2M1 code for each magnitude: how many midpoints it reaches."""
    code = Tensor.zeros_like(magnitude)
    for midpoint in _MIDPOINTS:
        code = code + (magnitude >= midpoint).cast(magnitude.dtype)
    return code


def _exact_data(rows: int, k: int, first: int, device: Any) -> Tensor:
    """Builds fp32 data that survives NVFP4 quantization unchanged.

    Values come straight off the level table, and any aligned run of 16
    covers every code, so each block's largest magnitude is 6 and its scale
    is exactly 1.0. Sums then stay multiples of 0.25 and well inside fp32's
    exact range, which keeps the pipeline independent of summation order --
    otherwise a host loop and a GPU launch could disagree.
    """
    idx = Tensor.arange(
        first, first + rows * k, dtype=DType.float32, device=device
    ).reshape([rows, k])
    sign = 1.0 - 2.0 * (idx % 3.0 == 0.0).cast(DType.float32)
    return _level(idx % 8.0) * sign


def _quantize_ref(x: Tensor) -> tuple[Tensor, Tensor]:
    """Mirrors `mxf353_nvfp4_quantize`: block scales and packed bytes.

    Comparing in packed form rather than unpacking a result keeps this to
    ops the framework already has.
    """
    rows, k = int(x.shape[0]), int(x.shape[1])
    blocks = k // _BLOCK
    absmax = abs(x).reshape([rows, blocks, _BLOCK]).max(axis=2)
    scales = absmax.reshape([rows, blocks]) / _NVFP4_MAX
    scales = scales + (scales == 0.0).cast(scales.dtype)
    per_element = (
        scales.reshape([rows, blocks, 1])
        .broadcast_to([rows, blocks, _BLOCK])
        .reshape([rows, k])
    )
    q = x / per_element
    code = _code(abs(q)) + 8.0 * (q < 0.0).cast(q.dtype)
    low, high = code.reshape([rows, k // 2, 2]).split([1, 1], axis=2)
    packed = low.reshape([rows, k // 2]) + 16.0 * high.reshape([rows, k // 2])
    return scales, packed


def _reference(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """The whole pipeline in framework ops: contract, relu, requantize.

    `x` and `w` quantize losslessly, so each stands in for its own
    dequantized value and the reference never unpacks anything.
    """
    acc = x @ w.T
    return _quantize_ref(acc * (acc > 0.0).cast(acc.dtype))


def _quantize_def() -> C.CustomOp:
    """Two outputs, changing both dtype and shape from the input."""
    rows, k = C.Symbols("rows", "k")
    return C.declare(
        "mxf353_nvfp4_quantize",
        inputs={"x": C.TemplateType(DType.float32, [rows, k])},
        outputs=[
            C.TemplateType(DType.uint8, [rows, k // 2]),
            C.TemplateType(DType.float32, [rows, k // _BLOCK]),
        ],
        custom_extensions=[KERNELS],
    )


def _matmul_def() -> C.CustomOp:
    m, n, kh, kb = C.Symbols("m", "n", "kh", "kb")
    return C.declare(
        "mxf353_nvfp4_matmul",
        inputs={
            "ap": C.TemplateType(DType.uint8, [m, kh]),
            "asc": C.TemplateType(DType.float32, [m, kb]),
            "bp": C.TemplateType(DType.uint8, [n, kh]),
            "bsc": C.TemplateType(DType.float32, [n, kb]),
        },
        outputs=[C.TemplateType(DType.float32, [m, n])],
        custom_extensions=[KERNELS],
    )


def _relu_def() -> C.CustomOp:
    rows, k = C.Symbols("rows", "k")
    return C.declare(
        "mxf353_relu",
        inputs={"x": C.TemplateType(DType.float32, [rows, k])},
        outputs=[C.TemplateType(DType.float32, [rows, k])],
        custom_extensions=[KERNELS],
    )


def _pipeline_eager(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """fp32 -> quantize -> nvfp4 matmul -> relu -> requantize, eagerly.

    One `quantize` def used three times: both operands and the activation.
    """
    quantize, matmul, relu = _quantize_def(), _matmul_def(), _relu_def()
    xq, xs = _two_tensors(quantize(x))
    wq, ws = _two_tensors(quantize(w))
    acc = _one_tensor(matmul(xq, xs, wq, ws))
    return _two_tensors(quantize(_one_tensor(relu(acc))))


def _pipeline_graph(x: Tensor, w: Tensor, device: Any) -> tuple[Tensor, Tensor]:
    """The same pipeline staged into one graph, compiled, and executed."""
    quantize, matmul, relu = _quantize_def(), _matmul_def(), _relu_def()
    dev = DeviceRef.from_device(device)
    with Graph(
        "g_nvfp4_pipeline",
        input_types=[
            TensorType(DType.float32, ["m", "k"], dev),
            TensorType(DType.float32, ["n", "k"], dev),
        ],
        custom_extensions=[KERNELS],
    ) as graph:
        xq, xs = _two_values(quantize(graph.inputs[0].tensor))
        wq, ws = _two_values(quantize(graph.inputs[1].tensor))
        acc = _one_value(matmul(xq, xs, wq, ws))
        packed, scales = _two_values(quantize(_one_value(relu(acc))))
        graph.output(packed, scales)

    compiled = engine.InferenceSession(devices=[device]).load(graph)
    outputs = compiled.execute(x.driver_tensor, w.driver_tensor)
    return tuple(Tensor(storage=buffer) for buffer in outputs)  # type: ignore[return-value]


def _assert_exact(want: Tensor, got: Tensor) -> None:
    """Asserts two tensors match exactly, at any rank.

    `assert_all_close` reduces with `max(axis=-1)`, so it only lands on a
    scalar for a rank-1 tensor; comparing flattened copies keeps any rank on
    that path.
    """
    assert [int(d) for d in want.shape] == [int(d) for d in got.shape]
    count = got.num_elements()
    assert_all_close(
        want.reshape([count]), got.reshape([count]), atol=0.0, rtol=0.0
    )


def _assert_quantization_equal(
    want: tuple[Tensor, Tensor], got: tuple[Tensor, Tensor]
) -> None:
    """Compares a (packed, scales) result exactly.

    The data is chosen so quantization is lossless, so anything but an exact
    match is a real disagreement rather than rounding.
    """
    want_scales, want_packed = want
    got_packed, got_scales = got
    _assert_exact(want_scales, got_scales)
    _assert_exact(want_packed, got_packed.cast(DType.float32))


@pytest.mark.parametrize("on_gpu", [False, True], ids=["cpu", "gpu"])
@pytest.mark.parametrize("staged", [False, True], ids=["eager", "graph"])
def test_nvfp4_pipeline_matches_reference(on_gpu: bool, staged: bool) -> None:
    """A four-op NVFP4 pipeline agrees with the framework's own ops on either
    device, eagerly and staged.

    Exercises a multi-output def reused at two points in one pipeline, a
    dtype change (fp32 to packed uint8), two shape changes (`k // 2` and
    `k // 16`), and an op taking four inputs.
    """
    device = Accelerator() if on_gpu else CPU()
    x = _exact_data(3, 32, 0, device)
    w = _exact_data(_N, 32, 101, device)

    if staged:
        got = _pipeline_graph(x, w, device)
    else:
        got = _pipeline_eager(x, w)

    assert [int(d) for d in got[0].shape] == [3, _N // 2]
    _assert_quantization_equal(_reference(x, w), got)


def test_same_def_serves_cpu_and_accelerator(
    force_interpreter_only: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One def called with host and device tensors compiles a binding per
    device and stays correct on both.

    ``device`` is part of the binding key, so the second call must miss the
    cache; reusing the first binding would run host code against device
    memory.
    """
    quantize = _quantize_def()
    compiles = _spy_on_compiles(monkeypatch)

    for expected_compiles, device in enumerate([CPU(), Accelerator()], start=1):
        x = _exact_data(2, 32, 5, device)
        got = _two_tensors(quantize(x))
        assert compiles[0] == expected_compiles
        _assert_quantization_equal(_quantize_ref(x), got)
    assert len(custom_gc._CACHE) == 2


def test_mixed_device_operands_raises() -> None:
    """Operands split across host and device are refused.

    One `mo.custom` carries one device attribute and the kernel receives one
    `DeviceContext`, so there is no single device to stage a mixed-device
    call onto. `_matmul_def`'s four operands make the split easy to
    construct: two land on the accelerator, two on the host.
    """
    matmul = _matmul_def()
    accel, cpu = Accelerator(), CPU()
    ap = Tensor.zeros([2, 4], dtype=DType.uint8, device=accel)
    asc = Tensor.zeros([2, 1], dtype=DType.float32, device=accel)
    bp = Tensor.zeros([2, 4], dtype=DType.uint8, device=cpu)
    bsc = Tensor.zeros([2, 1], dtype=DType.float32, device=cpu)

    with pytest.raises(ValueError, match="all inputs must be on one device"):
        matmul(ap, asc, bp, bsc)


def _stage_reserved_dim_on_accelerator() -> object:
    """Feeds an accelerator graph dim that imitates a signature symbol name
    to a def."""
    dev = DeviceRef.from_device(Accelerator())
    graph = Graph(
        "g_gpu_reserved_dim",
        input_types=[TensorType(DType.float32, ["__co_other_rows", 32], dev)],
        custom_extensions=[KERNELS],
    )
    with graph:
        return _quantize_def()(graph.inputs[0].tensor)


def test_graph_dim_imitating_signature_symbol_namespace_is_refused() -> None:
    """A graph dim in the namespace declared signature symbols live under
    is refused.

    Nothing stops a caller naming a dim this way, and if it reached staging
    it would be indistinguishable from a real signature symbol once bound,
    so the ambiguity is rejected at the boundary rather than resolved. A
    dim imitating the *allocated* data-dependent namespace instead is a
    different case and is deliberately not refused (see
    `test_custom_op_def.py`'s
    `test_data_dependent_dim_feeds_a_downstream_op_staged`).
    """
    with pytest.raises(TypeError, match="reserved"):
        _stage_reserved_dim_on_accelerator()
