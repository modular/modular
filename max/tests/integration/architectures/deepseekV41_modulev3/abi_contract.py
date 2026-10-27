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
"""Stages the V4.1 stub on SM100 virtual devices and records its input ABI.

Mach binds every graph input by position, so the contract it transcribes is
recorded from the graph the stub builds, not typed by hand. Staging reads no
weight bytes and needs no GPU.
"""

from __future__ import annotations

import json
import os
import platform
import re
import struct
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from f1_config import f1_config
from max import functional as F
from max.driver import (
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.graph.weights import load_weights
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    ExpertForm,
    ParamSpec,
    V41Config,
    build_kv_params,
    extra_inputs,
    input_names,
    kv_cache_config,
    parameter_specs,
)
from max.tensor import default_device
from max.tensor.compilation import StagedGraph
from stub import (
    DeepseekV41Stub,
    make_mesh,
)

TP_DEGREES = (1, 2, 4, 8)
# The recorded layouts. The F1 fixture's ABI is recorded once, in the
# engine-side golden the graph-contract recorder writes
# (mach/crates/engine-core/tests/testdata/v41_contract.json, in the M3
# recorder's format), so it is not duplicated here.
MODELS = ("full",)

# The census table's dtype spellings, as ``DType`` values.
CENSUS_DTYPES: Mapping[str, DType] = {
    "BF16": DType.bfloat16,
    "F32": DType.float32,
    "F8_E4M3": DType.float8_e4m3fn,
    "F8_E8M0": DType.float8_e8m0fnu,
    "U8": DType.uint8,
    "I8": DType.int8,
}

# Checkpoint tensors no parameter tree holds: the DSpark drafter, the
# vision tower, and Engram's tables, which are graph inputs.
_ALLOW_LISTED = re.compile(
    r"^(mtp\.|vision\.|aligner\.|image_(start|end|newline)$|"
    r"layers\.\d+\.engram\.embed\.(weight|scale)$)"
)


def allow_listed(name: str) -> bool:
    """Whether the census entry ``name`` is deliberately not tree-held."""
    return _ALLOW_LISTED.match(name) is not None


def _census_layers_for(config: V41Config, suffix: str) -> tuple[int, ...]:
    """The layers whose checkpoint holds ``layers.<L>.<suffix>``."""
    if suffix.startswith("engram."):
        return config.engram_layer_ids
    kv, ix = config.kv_source_layer_ids, config.index_source_layer_ids
    return {
        "attn.compressor.wkv.weight": kv,
        "attn.compressor.norm.weight": kv,
        "attn.compressor.wgate.weight": config.kv_layers_r2,
        "attn.indexer.wq_b.weight": ix,
        "attn.indexer.wq_b.scale": ix,
        "attn.indexer.weights_proj.weight": ix,
        "attn.indexer.wk.weight": kv,
        "attn.indexer.k_norm.weight": kv,
    }.get(suffix, tuple(range(config.num_hidden_layers)))


def _expand_census_pattern(config: V41Config, pattern: str) -> list[str]:
    """Expands one census name pattern to the names it stands for."""
    if pattern.startswith("mtp.{N}"):
        return []
    if "{L}" not in pattern:
        return [pattern]
    suffix = pattern.removeprefix("layers.{L}.")
    names = [
        pattern.replace("{L}", str(layer))
        for layer in _census_layers_for(config, suffix)
    ]
    if "{E}" in pattern:
        names = [
            name.replace("{E}", str(e))
            for name in names
            for e in range(config.n_routed_experts)
        ]
    return names


def census(
    config: V41Config,
) -> dict[str, tuple[DType, tuple[int, ...] | None]]:
    """The checkpoint's tensors, expanded from the census pattern table.

    The table is the whole checkpoint census in pattern form (its counts
    sum to the checkpoint's 188,245 tensor names), expanded to one entry
    per tensor with its dtype and shape. The shape is ``None`` for a
    pattern whose members differ in shape (Engram's tables).
    """
    out: dict[str, tuple[DType, tuple[int, ...] | None]] = {}
    text = (testdata_dir() / "census_patterns.tsv").read_text()
    for row in text.splitlines():
        count, _, pattern, dtype_col, shape_col = row.split("\t")
        names = _expand_census_pattern(config, pattern)
        if not pattern.startswith("mtp."):
            assert len(names) == int(count), (pattern, len(names), count)
        shapes = re.findall(r"\[([^\]]*)\]", shape_col)
        shape = (
            tuple(int(d) for d in shapes[0].split(",") if d.strip())
            if len(shapes) == 1
            else None
        )
        for name in names:
            out[name] = (CENSUS_DTYPES[dtype_col.split(":")[0]], shape)
    return out


def census_byte_extent(
    entry: tuple[DType, tuple[int, ...] | None],
) -> int | None:
    """One census entry's byte extent, or ``None`` if its shape varies."""
    dtype, shape = entry
    if shape is None:
        return None
    extent = dtype.size_in_bytes
    for dim in shape:
        extent *= dim
    return extent


def testdata_dir() -> Path:
    return Path(os.environ["PIPELINES_TESTDATA"])


def hf_config(model: str) -> Mapping[str, Any]:
    """The HF config of ``model``: ``"full"`` or ``"f1"``."""
    full = json.loads((testdata_dir() / "config_full.json").read_text())
    return full if model == "full" else f1_config(full)


@contextmanager
def virtual_devices(n_devices: int) -> Iterator[None]:
    """SM100 virtual devices: graph staging with no GPU attached.

    Must be entered before the process creates its first device.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_100a")
    set_virtual_device_count(n_devices)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def compile_input_types(
    config: V41Config,
    kv_params: MultiKVCacheParams,
    device_refs: Sequence[DeviceRef],
) -> list[TensorType | BufferType]:
    """The forward's input types in ABI order, before the signal buffers."""
    gpu0 = device_refs[0]
    return [
        TensorType(DType.int64, ["total_seq_len"], gpu0),
        TensorType(DType.int64, ["return_n_logits"], DeviceRef.CPU()),
        TensorType(DType.uint32, ["input_row_offsets_len"], gpu0),
        *kv_params.flattened_kv_inputs(),
        *(t for _, t in extra_inputs(config, device_refs)),
    ]


_SAFETENSORS_DTYPE = {
    DType.bfloat16: "BF16",
    DType.float32: "F32",
    DType.uint8: "U8",
    DType.float8_e4m3fn: "F8_E4M3",
    DType.float8_e8m0fnu: "F8_E8M0",
}


def write_sparse_safetensors(path: Path, specs: Sequence[ParamSpec]) -> None:
    """Writes a valid safetensors file whose data section is a hole.

    Staging and compiling read weight metadata, never the bytes, so the
    fixture costs no disk and no RAM.
    """
    header: dict[str, Any] = {}
    offset = 0
    for spec in specs:
        nbytes = spec.dtype.size_in_bytes
        for d in spec.shape:
            nbytes *= d
        header[spec.name] = {
            "dtype": _SAFETENSORS_DTYPE[spec.dtype],
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + offset)


def sparse_state_dict(
    config: V41Config, n_devices: int, experts: ExpertForm, workdir: Path
) -> dict[str, Any]:
    """The weights of the stub's whole tree, backed by a sparse file."""
    path = workdir / f"stub_tp{n_devices}_{experts}.safetensors"
    write_sparse_safetensors(path, parameter_specs(config, n_devices, experts))
    return {k: w.data() for k, w in load_weights([path]).items()}


def stage_stub(
    model: str, n_devices: int, experts: ExpertForm = "per_expert"
) -> tuple[StagedGraph[..., Any], V41Config, MultiKVCacheParams]:
    """Traces the stub of ``model`` at TP``n_devices``.

    Must run inside :func:`virtual_devices`.
    """
    config = V41Config.from_hf(hf_config(model))
    device_refs = [DeviceRef.GPU(i) for i in range(n_devices)]
    kv_params = build_kv_params(config, kv_cache_config(), device_refs)
    mesh = make_mesh(n_devices)
    with F.lazy(), default_device(mesh):
        stub = DeepseekV41Stub(config, kv_params, mesh, experts)
    staged = stub.trace(*compile_input_types(config, kv_params, device_refs))
    return staged, config, kv_params


def _device_label(ref: DeviceRef) -> str:
    return "cpu" if ref.device_type.value == "cpu" else f"gpu:{ref.id}"


def contract(
    staged: StagedGraph[..., Any],
    config: V41Config,
    kv_params: MultiKVCacheParams,
    n_devices: int,
) -> list[dict[str, Any]]:
    """One record per graph input: slot, name, kind, dtype, shape, device."""
    names = input_names(config, kv_params, n_devices)
    types = [value.type for value in staged.graph.inputs]
    assert len(names) == len(types), (len(names), len(types))
    records = []
    for i, (name, t) in enumerate(zip(names, types, strict=True)):
        assert isinstance(t, (TensorType, BufferType))
        records.append(
            {
                "index": i,
                "name": name,
                "kind": "buffer" if isinstance(t, BufferType) else "tensor",
                "dtype": str(t.dtype),
                "symbolic_shape": [
                    int(d) if str(d).isdigit() else str(d) for d in t.shape
                ],
                "device": _device_label(t.device),
            }
        )
    return records


COMMENT = (
    "Input slots of the DeepSeek-V4.1-Flash ModuleV3 stub graph, staged in "
    "virtual-device mode (cuda sm_100a). Regenerate with "
    "//max/tests/integration/architectures/deepseekV41_modulev3:"
    "record_abi_contract, never by hand. The F1 fixture's ABI is recorded "
    "separately, in mach/crates/engine-core/tests/testdata/"
    "v41_contract.json, which the graph-contract recorder writes in the M3 "
    "format the engine's contract test reads."
)


def record() -> dict[str, Any]:
    """The full model's contract at every TP degree."""
    layouts: dict[str, Any] = {}
    for model in MODELS:
        for n in TP_DEGREES:
            with virtual_devices(n):
                staged, config, kv_params = stage_stub(model, n)
                layouts[f"{model}_tp{n}"] = contract(
                    staged, config, kv_params, n
                )
    return {"comment": COMMENT, "layouts": layouts}


def dumps(data: Mapping[str, Any]) -> str:
    """Serializes :func:`record`'s output with one input record per line."""
    lines = [
        "{",
        f' "comment": {json.dumps(data["comment"])},',
        ' "layouts": {',
    ]
    layouts = list(data["layouts"].items())
    for i, (key, records) in enumerate(layouts):
        lines.append(f"  {json.dumps(key)}: [")
        lines += [
            f"   {json.dumps(r)}{',' if j < len(records) - 1 else ''}"
            for j, r in enumerate(records)
        ]
        lines.append(f"  ]{',' if i < len(layouts) - 1 else ''}")
    lines += [" }", "}"]
    return "\n".join(lines) + "\n"


def run_cell(
    model: str,
    n_devices: int,
    experts: ExpertForm = "per_expert",
    *,
    compile_graph: bool = True,
    count_bodies: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Builds, stages and optionally compiles one cell; returns its timings.

    Must run in a fresh process. Compiling needs a Linux host: the NVPTX
    target is not available on macOS. The second result is the names of the
    compiled weight registry (empty unless ``compile_graph``).
    """
    result: dict[str, Any] = {
        "config": model,
        "tp": n_devices,
        "experts": experts,
        "machine": platform.platform(),
    }
    registry: list[str] = []
    with virtual_devices(n_devices), tempfile.TemporaryDirectory() as workdir:
        config = V41Config.from_hf(hf_config(model))
        device_refs = [DeviceRef.GPU(i) for i in range(n_devices)]
        kv_params = build_kv_params(config, kv_cache_config(), device_refs)
        mesh = make_mesh(n_devices)
        input_types = compile_input_types(config, kv_params, device_refs)

        t0 = time.perf_counter()
        with F.lazy(), default_device(mesh):
            stub = DeepseekV41Stub(config, kv_params, mesh, experts)
        result["build_s"] = time.perf_counter() - t0
        result["parameters"] = sum(1 for _ in stub.parameters)

        t0 = time.perf_counter()
        staged = stub.trace(*input_types)
        result["stage_s"] = time.perf_counter() - t0
        result["graph_inputs"] = len(staged.graph.inputs)
        if count_bodies:
            bodies = re.findall(r"mo.graph @(\w+)", str(staged))
            result["subgraphs"] = bodies[1:]

        if compile_graph:
            state_dict = sparse_state_dict(
                config, n_devices, experts, Path(workdir)
            )
            t0 = time.perf_counter()
            compiled = stub.compile(*input_types, weights=state_dict)
            result["compile_total_s"] = time.perf_counter() - t0
            registry = list(compiled.weights)
            result["registry_entries"] = len(registry)
    return result, registry
