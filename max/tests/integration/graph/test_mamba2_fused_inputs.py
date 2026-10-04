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
"""The Mamba-2 conv and SSD scan ops read sliced operands without copying them.

The conv input and the SSD scan's ``x``, ``dt``, ``B``, ``C`` and pool rows are
fused inputs. A slice feeding one of them fuses into it, and the op reads the
slice's storage in place through its strides. This runs the ops at
Nemotron-3.5-Lightning's Mamba dimensions on identical data:

* fused: slice the operands out of the in-projection and conv outputs, and
  the layer's ``[1, batch]`` pool rows out of every layer's, inside the graph.
* plain: hand the ops the same operands as contiguous graph inputs.

The kernels see the same values either way, so outputs and state pools must be
bitwise equal.
"""

from __future__ import annotations

from collections import Counter

import max.driver as md
import numpy as np
import pytest
import torch
from max.driver import accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.state_space import (
    causal_conv1d_varlen_fwd,
    mamba2_ssd_chunk_scan_varlen_fwd_inplace,
)
from torch.profiler import ProfilerActivity, profile

# Nemotron-3.5-Lightning's Mamba mixer.
_HEADS, _HEAD_DIM = 64, 64
_GROUPS, _STATE = 8, 128
_KERNEL = 4
_INTERMEDIATE = _HEADS * _HEAD_DIM
_GROUP_DIM = _GROUPS * _STATE
_CONV_DIM = _INTERMEDIATE + 2 * _GROUP_DIM
_PROJ_WIDTH = _INTERMEDIATE + _CONV_DIM + _HEADS
_NUM_LAYERS = 3
_LAYER = 1

_BF16 = DType.bfloat16
_GPU = DeviceRef.GPU()
_CPU = DeviceRef.CPU()
_LAYER_TYPE = TensorType(DType.int64, [], device=_CPU)


def _conv_pool() -> BufferType:
    return BufferType(_BF16, ["rows", _CONV_DIM, _KERNEL - 1], device=_GPU)


def _ssm_pool() -> BufferType:
    return BufferType(
        DType.float32, ["rows", _HEADS, _HEAD_DIM, _STATE], device=_GPU
    )


def _layer_rows(table: TensorValue, layer: TensorValue) -> TensorValue:
    """Row ``layer`` of a ``[num_layers, batch]`` table, as ``[1, batch]``."""
    return table[(slice(layer, layer + 1), 1), :]


def _conv(
    x: TensorValue,
    weight: TensorValue,
    bias: TensorValue,
    pool: BufferValue,
    qsl: TensorValue,
    rows: TensorValue,
    has_initial_state: TensorValue,
) -> TensorValue:
    return causal_conv1d_varlen_fwd(
        x=x,
        weight=weight,
        bias=bias,
        conv_states=pool,
        query_start_loc=qsl,
        cache_indices=rows,
        has_initial_state=has_initial_state,
        activation="silu",
        channels_last=True,
    )


def _ssd_operands(
    proj: TensorValue, xbc: TensorValue
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
    """``dt``, ``x``, ``B`` and ``C``, sliced out of ``proj`` and ``xbc``."""
    return (
        proj[:, _INTERMEDIATE + _CONV_DIM :],
        xbc[:, :_INTERMEDIATE],
        xbc[:, _INTERMEDIATE : _INTERMEDIATE + _GROUP_DIM],
        xbc[:, _INTERMEDIATE + _GROUP_DIM :],
    )


def _ssd(
    operands: tuple[TensorValue, TensorValue, TensorValue, TensorValue],
    params: list[TensorValue],
    pool: BufferValue,
    rows: TensorValue,
) -> TensorValue:
    """The SSD scan on ``(dt, x, B, C)``; ``params`` holds ``A``, ``D``,
    ``dt_bias``, the start offsets and ``has_initial_state``."""
    dt, x, B, C = operands
    A, D, dt_bias, qsl, has_initial_state = params
    return mamba2_ssd_chunk_scan_varlen_fwd_inplace(
        x=x.reshape([-1, _HEADS, _HEAD_DIM]),
        dt=dt,
        A=A,
        B=B.reshape([-1, _GROUPS, _STATE]),
        C=C.reshape([-1, _GROUPS, _STATE]),
        D=D,
        dt_bias=dt_bias,
        ssm_pool=pool,
        query_start_loc=qsl,
        has_initial_state=has_initial_state,
        cache_indices=rows,
    )


def _conv_graph(fused: bool) -> Graph:
    """The conv op on a slice of ``proj`` (fused) or on a contiguous input."""
    x_type = TensorType(
        _BF16, ["n", _PROJ_WIDTH if fused else _CONV_DIM], device=_GPU
    )
    rows_type = TensorType(
        DType.uint32,
        [_NUM_LAYERS, "batch"] if fused else ["batch"],
        device=_GPU,
    )
    with Graph(
        "conv_fused" if fused else "conv_plain",
        input_types=[
            x_type,
            TensorType(_BF16, [_CONV_DIM, _KERNEL], device=_GPU),
            TensorType(_BF16, [_CONV_DIM], device=_GPU),
            _conv_pool(),
            TensorType(DType.int32, ["batch_plus_one"], device=_GPU),
            rows_type,
            TensorType(DType.bool, ["batch"], device=_GPU),
            _LAYER_TYPE,
        ],
    ) as graph:
        x, weight, bias, pool, qsl, rows, has_initial_state, layer = (
            graph.inputs
        )
        x_value = x.tensor
        rows_value = rows.tensor
        if fused:
            x_value = x_value[:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM]
            rows_value = _layer_rows(rows_value, layer.tensor)
        graph.output(
            _conv(
                x_value,
                weight.tensor,
                bias.tensor,
                pool.buffer,
                qsl.tensor,
                rows_value,
                has_initial_state.tensor,
            )
        )
    return graph


def _ssd_input_types(fused: bool) -> list[TensorType | BufferType]:
    if fused:
        operand_types = [
            TensorType(_BF16, ["n", _PROJ_WIDTH], device=_GPU),
            TensorType(_BF16, ["n", _CONV_DIM], device=_GPU),
        ]
    else:
        operand_types = [
            TensorType(_BF16, ["n", _HEADS], device=_GPU),
            TensorType(_BF16, ["n", _INTERMEDIATE], device=_GPU),
            TensorType(_BF16, ["n", _GROUP_DIM], device=_GPU),
            TensorType(_BF16, ["n", _GROUP_DIM], device=_GPU),
        ]
    return [
        *operand_types,
        TensorType(_BF16, [_HEADS], device=_GPU),
        TensorType(_BF16, [_HEADS], device=_GPU),
        TensorType(_BF16, [_HEADS], device=_GPU),
        _ssm_pool(),
        TensorType(DType.int32, ["batch_plus_one"], device=_GPU),
        TensorType(DType.bool, ["batch"], device=_GPU),
        TensorType(
            DType.uint32,
            [_NUM_LAYERS, "batch"] if fused else ["batch"],
            device=_GPU,
        ),
        _LAYER_TYPE,
    ]


def _ssd_graph(fused: bool) -> Graph:
    """The SSD scan on slices of ``proj`` and ``xbc`` (fused) or on
    contiguous inputs."""
    num_operands = 2 if fused else 4
    with Graph(
        "ssd_fused" if fused else "ssd_plain",
        input_types=_ssd_input_types(fused),
    ) as graph:
        inputs = graph.inputs
        operands = [v.tensor for v in inputs[:num_operands]]
        A, D, dt_bias, pool, qsl, has_initial_state, rows, layer = inputs[
            num_operands:
        ]
        params = [
            A.tensor,
            D.tensor,
            dt_bias.tensor,
            qsl.tensor,
            has_initial_state.tensor,
        ]
        if fused:
            proj, xbc = operands
            y = _ssd(
                _ssd_operands(proj, xbc),
                params,
                pool.buffer,
                _layer_rows(rows.tensor, layer.tensor),
            )
        else:
            dt, x, B, C = operands
            y = _ssd((dt, x, B, C), params, pool.buffer, rows.tensor)
        graph.output(y)
    return graph


def _to_device(
    tensors: list[torch.Tensor], devices: list[md.Device]
) -> list[md.Buffer]:
    return [
        md.Buffer.from_dlpack(t.clone().contiguous()).to(device)
        for t, device in zip(tensors, devices, strict=True)
    ]


def _run(
    model: Model, tensors: list[torch.Tensor], pool_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs a graph on copies of ``tensors``; returns its output and pool."""
    buffers = _to_device(tensors, model.input_devices)
    (output,) = model.execute(*buffers)
    return torch.from_dlpack(output).cpu(), torch.from_dlpack(
        buffers[pool_index]
    ).cpu()


def _random_state(seq_lens: list[int], seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    batch = len(seq_lens)
    num_rows = _NUM_LAYERS * batch + 4

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (scale * torch.randn(shape, generator=gen)).to(torch.bfloat16)

    def rows() -> torch.Tensor:
        ids = torch.randperm(num_rows, generator=gen)[: _NUM_LAYERS * batch]
        return ids.reshape(_NUM_LAYERS, batch).to(torch.uint32)

    return {
        "proj": randn(sum(seq_lens), _PROJ_WIDTH),
        "conv_weight": randn(_CONV_DIM, _KERNEL, scale=0.3),
        "conv_bias": randn(_CONV_DIM, scale=0.3),
        "conv_pool": randn(num_rows, _CONV_DIM, _KERNEL - 1),
        "query_start_loc": torch.tensor(
            np.cumsum([0, *seq_lens]), dtype=torch.int32
        ),
        "conv_rows": rows(),
        "has_initial_state": torch.ones(batch, dtype=torch.bool),
        "A": -torch.exp(torch.rand(_HEADS, generator=gen) * 2).to(
            torch.bfloat16
        ),
        "D": randn(_HEADS, scale=0.3),
        "dt_bias": randn(_HEADS, scale=0.5) - 2,
        "ssm_pool": 0.1
        * torch.randn([num_rows, _HEADS, _HEAD_DIM, _STATE], generator=gen),
        "ssm_rows": rows(),
        "layer": torch.tensor(_LAYER, dtype=torch.int64),
    }


_SEQ_LENS = [[1] * 64, [1], [5, 1, 9], [300, 1, 40, 1, 700]]
_IDS = ["decode64", "decode1", "mixed", "prefill"]


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize("seq_lens", _SEQ_LENS, ids=_IDS)
def test_fused_inputs_match_plain_inputs(
    session: InferenceSession, seq_lens: list[int]
) -> None:
    s = _random_state(seq_lens, seed=len(seq_lens))
    conv_fused = session.load(_conv_graph(fused=True))
    conv_plain = session.load(_conv_graph(fused=False))
    ssd_fused = session.load(_ssd_graph(fused=True))
    ssd_plain = session.load(_ssd_graph(fused=False))

    # Conv: a column range of ``proj`` against the same columns made contiguous.
    conv_common = [s["conv_weight"], s["conv_bias"], s["conv_pool"]]
    xbc_fused, conv_pool_fused = _run(
        conv_fused,
        [
            s["proj"],
            *conv_common,
            s["query_start_loc"],
            s["conv_rows"],
            s["has_initial_state"],
            s["layer"],
        ],
        pool_index=3,
    )
    xbc_plain, conv_pool_plain = _run(
        conv_plain,
        [
            s["proj"][:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM],
            *conv_common,
            s["query_start_loc"],
            s["conv_rows"][_LAYER],
            s["has_initial_state"],
            s["layer"],
        ],
        pool_index=3,
    )
    assert torch.isfinite(xbc_fused.float()).all()
    assert torch.equal(xbc_fused, xbc_plain)
    assert torch.equal(conv_pool_fused, conv_pool_plain)

    # SSD scan: the operands of one conv output against contiguous copies.
    ssd_common = [s["A"], s["D"], s["dt_bias"], s["ssm_pool"]]
    tail = [s["query_start_loc"], s["has_initial_state"]]
    y_fused, ssm_pool_fused = _run(
        ssd_fused,
        [s["proj"], xbc_plain, *ssd_common, *tail, s["ssm_rows"], s["layer"]],
        pool_index=5,
    )
    dt = s["proj"][:, _INTERMEDIATE + _CONV_DIM :]
    y_plain, ssm_pool_plain = _run(
        ssd_plain,
        [
            dt,
            xbc_plain[:, :_INTERMEDIATE],
            xbc_plain[:, _INTERMEDIATE : _INTERMEDIATE + _GROUP_DIM],
            xbc_plain[:, _INTERMEDIATE + _GROUP_DIM :],
            *ssd_common,
            *tail,
            s["ssm_rows"][_LAYER],
            s["layer"],
        ],
        pool_index=7,
    )
    assert torch.isfinite(y_fused.float()).all()
    assert torch.equal(y_fused, y_plain)
    assert torch.equal(ssm_pool_fused, ssm_pool_plain)

    # Only the calling layer's rows changed.
    for after, before, rows in (
        (conv_pool_fused, s["conv_pool"], s["conv_rows"]),
        (ssm_pool_fused, s["ssm_pool"], s["ssm_rows"]),
    ):
        used = set(rows[_LAYER].tolist())
        for row in range(before.shape[0]):
            assert torch.equal(after[row], before[row]) == (row not in used)


def _launches(model: Model, buffers: list[md.Buffer]) -> Counter[str]:
    """The GPU kernels a few runs of ``model`` launch."""
    model.execute(*buffers)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            model.execute(*buffers)
        md.Accelerator().synchronize()
    return Counter(
        e.name
        for e in prof.events()
        if e.device_type.name == "CUDA" and "Memcpy" not in e.name
    )


def _copies(launches: Counter[str]) -> list[str]:
    return [
        name
        for name in launches
        if "causal_conv" not in name and "mamba2_ssd" not in name
    ]


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_fused_inputs_launch_no_copies(session: InferenceSession) -> None:
    """Neither the operand slices nor the pool rows are copied."""
    s = _random_state([1] * 64, seed=7)
    ssd = session.load(_ssd_graph(fused=True))
    conv = session.load(_conv_graph(fused=True))
    ssd_inputs = [
        s["proj"],
        torch.zeros(64, _CONV_DIM, dtype=torch.bfloat16),
        s["A"],
        s["D"],
        s["dt_bias"],
        s["ssm_pool"],
        s["query_start_loc"],
        s["has_initial_state"],
        s["ssm_rows"],
        s["layer"],
    ]
    conv_inputs = [
        s["proj"],
        s["conv_weight"],
        s["conv_bias"],
        s["conv_pool"],
        s["query_start_loc"],
        s["conv_rows"],
        s["has_initial_state"],
        s["layer"],
    ]
    launches: Counter[str] = Counter()
    for model, tensors in ((ssd, ssd_inputs), (conv, conv_inputs)):
        launches += _launches(model, _to_device(tensors, model.input_devices))
    assert not _copies(launches), launches
    assert any("causal_conv" in name for name in launches)
    assert any("mamba2_ssd" in name for name in launches)


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_computed_operand_matches_materialized(
    session: InferenceSession,
) -> None:
    """An operand the graph computes rather than slices still reads right.

    An ``abs`` fuses into the SSD scan's ``dt`` input, which the kernel reads
    through its fused input with no storage behind it, so outputs and state pool
    must match a run on the materialized ``abs``.
    """
    s = _random_state([5, 1, 9], seed=3)
    dt = s["proj"][:, _INTERMEDIATE + _CONV_DIM :]
    tensors = [
        dt,
        s["proj"][:, :_INTERMEDIATE],
        s["proj"][:, _INTERMEDIATE : _INTERMEDIATE + _GROUP_DIM],
        s["proj"][
            :, _INTERMEDIATE + _GROUP_DIM : _INTERMEDIATE + 2 * _GROUP_DIM
        ],
        s["A"],
        s["D"],
        s["dt_bias"],
        s["ssm_pool"],
        s["query_start_loc"],
        s["has_initial_state"],
        s["ssm_rows"][_LAYER],
        s["layer"],
    ]
    plain = session.load(_ssd_graph(fused=False))
    with Graph("ssd_computed_dt", input_types=_ssd_input_types(False)) as graph:
        inputs = graph.inputs
        dt_in, x, B, C, A, D, dt_bias = (v.tensor for v in inputs[:7])
        params = [A, D, dt_bias, inputs[8].tensor, inputs[9].tensor]
        graph.output(
            _ssd(
                (ops.abs(dt_in), x, B, C),
                params,
                inputs[7].buffer,
                inputs[10].tensor,
            )
        )
    computed = session.load(graph)
    y_computed, pool_computed = _run(computed, tensors, pool_index=7)
    y_plain, pool_plain = _run(plain, [dt.abs(), *tensors[1:]], pool_index=7)
    assert torch.equal(y_computed, y_plain)
    assert torch.equal(pool_computed, pool_plain)


def _layers_graph(rows_in_subgraph: bool) -> Graph:
    """Two Mamba layers calling one shared subgraph, as Nemotron-H does.

    The caller either slices each layer's pool rows out of the
    ``[num_layers, batch]`` row tables, or hands the subgraph the tables and
    the layer, and the subgraph slices its ``[1, batch]`` rows itself.
    """
    table_type = TensorType(DType.uint32, [_NUM_LAYERS, "batch"], device=_GPU)
    row_type = TensorType(DType.uint32, ["batch"], device=_GPU)
    common_types: list[TensorType | BufferType] = [
        TensorType(_BF16, ["n", _PROJ_WIDTH], device=_GPU),
        TensorType(_BF16, [_CONV_DIM, _KERNEL], device=_GPU),
        TensorType(_BF16, [_CONV_DIM], device=_GPU),
        _conv_pool(),
        TensorType(DType.int32, ["batch_plus_one"], device=_GPU),
        TensorType(DType.bool, ["batch"], device=_GPU),
        TensorType(_BF16, [_HEADS], device=_GPU),
        TensorType(_BF16, [_HEADS], device=_GPU),
        TensorType(_BF16, [_HEADS], device=_GPU),
        _ssm_pool(),
    ]
    row_types: list[TensorType] = (
        [table_type, table_type, _LAYER_TYPE]
        if rows_in_subgraph
        else [row_type, row_type]
    )
    with Graph(
        "layers_rows_inside" if rows_in_subgraph else "layers_rows_outside",
        input_types=[*common_types, table_type, table_type],
    ) as graph:
        with graph.add_subgraph(
            "mamba", input_types=[*common_types, *row_types]
        ) as sub:
            proj, weight, bias, conv_pool, qsl, has_initial_state = sub.inputs[
                :6
            ]
            A, D, dt_bias, ssm_pool = sub.inputs[6:10]
            if rows_in_subgraph:
                conv_table, ssm_table, layer = (
                    v.tensor for v in sub.inputs[10:]
                )
                conv_rows = _layer_rows(conv_table, layer)
                ssm_rows = _layer_rows(ssm_table, layer)
            else:
                conv_rows, ssm_rows = (v.tensor for v in sub.inputs[10:])
            xbc = _conv(
                proj.tensor[:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM],
                weight.tensor,
                bias.tensor,
                conv_pool.buffer,
                qsl.tensor,
                conv_rows,
                has_initial_state.tensor,
            )
            params = [
                A.tensor,
                D.tensor,
                dt_bias.tensor,
                qsl.tensor,
                has_initial_state.tensor,
            ]
            sub.output(
                _ssd(
                    _ssd_operands(proj.tensor, xbc),
                    params,
                    ssm_pool.buffer,
                    ssm_rows,
                )
            )
        common = graph.inputs[: len(common_types)]
        conv_table, ssm_table = (v.tensor for v in graph.inputs[-2:])
        outputs = []
        for index in range(2):
            rows = (
                [conv_table, ssm_table, ops.constant(index, DType.int64, _CPU)]
                if rows_in_subgraph
                else [conv_table[index], ssm_table[index]]
            )
            outputs += ops.call(sub, *common, *rows)
        graph.output(*outputs)
    return graph


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_layer_rows_sliced_in_subgraph(session: InferenceSession) -> None:
    """Slicing a layer's pool rows inside the shared subgraph copies nothing.

    Both graphs run the same two layers on identical data, so outputs and
    state pools must be bitwise equal. Sliced by the caller, each layer's two
    rows are copied to cross into the subgraph; sliced inside it, they are
    read in place and only the two ops launch.
    """
    s = _random_state([1] * 64, seed=11)
    tensors = [
        s["proj"],
        s["conv_weight"],
        s["conv_bias"],
        s["conv_pool"],
        s["query_start_loc"],
        s["has_initial_state"],
        s["A"],
        s["D"],
        s["dt_bias"],
        s["ssm_pool"],
        s["conv_rows"],
        s["ssm_rows"],
    ]
    results = {}
    launches = {}
    for inside in (False, True):
        model = session.load(_layers_graph(inside))
        buffers = _to_device(tensors, model.input_devices)
        outputs = model.execute(*buffers)
        results[inside] = [torch.from_dlpack(o).cpu() for o in outputs] + [
            torch.from_dlpack(buffers[i]).cpu() for i in (3, 9)
        ]
        launches[inside] = _launches(model, buffers)
    for fused, plain in zip(results[True], results[False], strict=True):
        assert torch.equal(fused, plain)
    assert not _copies(launches[True]), launches[True]
    assert _copies(launches[False]), launches[False]


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_unsliced_layer_rows_are_rejected(session: InferenceSession) -> None:
    """A ``[num_layers, batch]`` row table is not silently read as row 0."""
    s = _random_state([1] * 4, seed=5)
    with Graph(
        "conv_rows_table",
        input_types=[
            TensorType(_BF16, ["n", _CONV_DIM], device=_GPU),
            TensorType(_BF16, [_CONV_DIM, _KERNEL], device=_GPU),
            TensorType(_BF16, [_CONV_DIM], device=_GPU),
            _conv_pool(),
            TensorType(DType.int32, ["batch_plus_one"], device=_GPU),
            TensorType(DType.uint32, [_NUM_LAYERS, "batch"], device=_GPU),
            TensorType(DType.bool, ["batch"], device=_GPU),
        ],
    ) as graph:
        x, weight, bias, pool, qsl, rows, has_initial_state = graph.inputs
        graph.output(
            _conv(
                x.tensor,
                weight.tensor,
                bias.tensor,
                pool.buffer,
                qsl.tensor,
                rows.tensor,
                has_initial_state.tensor,
            )
        )
    model = session.load(graph)
    buffers = _to_device(
        [
            s["proj"][:, _INTERMEDIATE : _INTERMEDIATE + _CONV_DIM],
            s["conv_weight"],
            s["conv_bias"],
            s["conv_pool"],
            s["query_start_loc"],
            s["conv_rows"],
            s["has_initial_state"],
        ],
        model.input_devices,
    )
    with pytest.raises(Exception, match="cache_indices"):
        model.execute(*buffers)


_STRIDED = ["transposed", "stepped"]


def _storage_type(tail: list[int], kind: str) -> TensorType:
    """The type of the storage behind an ``[n, *tail]`` operand."""
    if kind == "transposed":
        return TensorType(_BF16, [*reversed(tail), "n"], device=_GPU)
    return TensorType(_BF16, ["n", *tail[:-1], 2 * tail[-1]], device=_GPU)


def _storage(operand: torch.Tensor, kind: str) -> torch.Tensor:
    """Storage that ``_view`` turns into ``operand`` with a last dim that is
    not contiguous. A stepped storage holds garbage between the elements."""
    if kind == "transposed":
        return operand.permute(*reversed(range(operand.ndim))).contiguous()
    shape = [*operand.shape[:-1], 2 * operand.shape[-1]]
    storage = torch.randn(shape).to(torch.bfloat16)
    storage[..., ::2] = operand
    return storage


def _view(storage: TensorValue, kind: str) -> TensorValue:
    if kind == "transposed":
        return ops.permute(storage, list(reversed(range(storage.rank))))
    return storage[(*[slice(None)] * (storage.rank - 1), slice(None, None, 2))]


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
@pytest.mark.parametrize("kind", _STRIDED)
@pytest.mark.parametrize("seq_lens", [[1] * 8, [5, 1, 9, 300]], ids=["d", "p"])
def test_non_contiguous_operands_match_contiguous(
    session: InferenceSession, kind: str, seq_lens: list[int]
) -> None:
    """Operands whose last dim is not contiguous still read right.

    ``dt``, ``x``, ``B``, ``C`` and the conv input are transposed or
    stepped views, so a vector load of a few elements of one row cannot be a
    contiguous read. Outputs and state pools must equal a run on contiguous
    copies.
    """
    s = _random_state(seq_lens, seed=17)
    n = sum(seq_lens)
    dims = {
        "dt": [_HEADS],
        "x": [_HEADS, _HEAD_DIM],
        "B": [_GROUPS, _STATE],
        "C": [_GROUPS, _STATE],
    }
    gen = torch.Generator().manual_seed(23)
    operands = {
        name: torch.randn(n, *tail, generator=gen).to(torch.bfloat16)
        for name, tail in dims.items()
    }

    def ssd_graph(strided: bool) -> Graph:
        types: list[TensorType | BufferType] = [
            _storage_type(tail, kind)
            if strided
            else TensorType(_BF16, ["n", *tail], device=_GPU)
            for tail in dims.values()
        ]
        types += [
            TensorType(_BF16, [_HEADS], device=_GPU),
            TensorType(_BF16, [_HEADS], device=_GPU),
            TensorType(_BF16, [_HEADS], device=_GPU),
            _ssm_pool(),
            TensorType(DType.int32, ["batch_plus_one"], device=_GPU),
            TensorType(DType.bool, ["batch"], device=_GPU),
            TensorType(DType.uint32, ["batch"], device=_GPU),
        ]
        with Graph(
            "ssd_strided" if strided else "ssd_contiguous", input_types=types
        ) as graph:
            dt, x, B, C, A, D, dt_bias, pool, qsl, his, rows = graph.inputs
            ops_in = [
                _view(v.tensor, kind) if strided else v.tensor
                for v in (dt, x, B, C)
            ]
            graph.output(
                _ssd(
                    (ops_in[0], ops_in[1], ops_in[2], ops_in[3]),
                    [
                        A.tensor,
                        D.tensor,
                        dt_bias.tensor,
                        qsl.tensor,
                        his.tensor,
                    ],
                    pool.buffer,
                    rows.tensor,
                )
            )
        return graph

    tail = [
        s["A"],
        s["D"],
        s["dt_bias"],
        s["ssm_pool"],
        s["query_start_loc"],
        s["has_initial_state"],
        s["ssm_rows"][_LAYER],
    ]
    strided = session.load(ssd_graph(True))
    plain = session.load(ssd_graph(False))
    y_s, pool_s = _run(
        strided,
        [_storage(operands[k], kind) for k in dims] + tail,
        pool_index=7,
    )
    y_p, pool_p = _run(plain, [operands[k] for k in dims] + tail, pool_index=7)
    assert torch.isfinite(y_s.float()).all()
    assert torch.equal(y_s, y_p)
    assert torch.equal(pool_s, pool_p)

    # Conv input.
    conv_x = torch.randn(n, _CONV_DIM, generator=gen).to(torch.bfloat16)

    def conv_graph(strided: bool) -> Graph:
        x_type = (
            _storage_type([_CONV_DIM], kind)
            if strided
            else TensorType(_BF16, ["n", _CONV_DIM], device=_GPU)
        )
        with Graph(
            "conv_strided" if strided else "conv_contiguous",
            input_types=[
                x_type,
                TensorType(_BF16, [_CONV_DIM, _KERNEL], device=_GPU),
                TensorType(_BF16, [_CONV_DIM], device=_GPU),
                _conv_pool(),
                TensorType(DType.int32, ["batch_plus_one"], device=_GPU),
                TensorType(DType.uint32, ["batch"], device=_GPU),
                TensorType(DType.bool, ["batch"], device=_GPU),
            ],
        ) as graph:
            x, w, b, pool, qsl, rows, his = graph.inputs
            graph.output(
                _conv(
                    _view(x.tensor, kind) if strided else x.tensor,
                    w.tensor,
                    b.tensor,
                    pool.buffer,
                    qsl.tensor,
                    rows.tensor,
                    his.tensor,
                )
            )
        return graph

    conv_tail = [
        s["conv_weight"],
        s["conv_bias"],
        s["conv_pool"],
        s["query_start_loc"],
        s["conv_rows"][_LAYER],
        s["has_initial_state"],
    ]
    xs, cp_s = _run(
        session.load(conv_graph(True)),
        [_storage(conv_x, kind), *conv_tail],
        pool_index=3,
    )
    xp, cp_p = _run(
        session.load(conv_graph(False)), [conv_x, *conv_tail], pool_index=3
    )
    assert torch.equal(xs, xp)
    assert torch.equal(cp_s, cp_p)
