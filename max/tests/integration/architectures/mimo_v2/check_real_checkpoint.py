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
"""Checks the MiMo-V2 weight adapter on the real checkpoints, CPU only.

Needs both snapshots (``ProCreations/MiMo-V2.6-Flash-RL-NVFP4`` and
``XiaomiMiMo/MiMo-V2.6-Flash-RL``) and about 170 GB of free RAM:

    CUDA_VISIBLE_DEVICES= python check_real_checkpoint.py \\
        --nvfp4 <snapshot> --upstream <snapshot> --out result.json

It first injects defects that must fail the load into in-memory copies and
requires each to raise. It then adapts the whole export through
``convert_safetensor_state_dict`` and checks the result against bytes the
adapter never read: every expert slice of the stacked tensors against
Xiaomi's MXFP4 tensor, every FP8 dense tensor against Xiaomi's FP8 codes and
scales and against the export's F32 after dequantization, and the router
against its BF16 source. It records a SHA-256 for every adapted tensor,
grouped by class, and for every expert's E8M0 slice, keyed by the checkpoint
name of the scale it came from, for comparison with Mach's Rust adapter.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import mmap
import multiprocessing
import re
import struct
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import SafetensorWeights, WeightData, Weights
from max.pipelines.architectures.mimo_v2.weight_adapters import (
    convert_safetensor_state_dict,
    e8m0_scales_from_nvfp4,
    fp8_block_scaled_from_float32,
    qkv_chunk_layout,
    qkv_to_chunk_order,
)
from transformers.configuration_utils import PretrainedConfig

# Xiaomi's fused qkv_proj: 4 chunks of (q, k, v) rows, fixed by the checkpoint.
_CHUNKS = 4
_FULL = (3072, 192, 128)
_SLIDING = (3072, 384, 256)
_STACK = re.compile(
    r"^layers\.(\d+)\.mlp\.experts\.(gate_up|down)_proj(_scale)?$"
)


class RawShards:
    """A safetensors reader that shares no code with MAX's loader."""

    def __init__(self, root: Path) -> None:
        index = json.loads((root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]
        self.root = root
        self.files: dict[str, tuple[dict[str, Any], int, mmap.mmap]] = {}

    def array(self, name: str, dtype: type[np.generic]) -> np.ndarray:
        shard = self.weight_map[name]
        if shard not in self.files:
            f = open(self.root / shard, "rb")
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
            data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.files[shard] = (header, 8 + size, data)
        header, base, data = self.files[shard]
        start, end = header[name]["data_offsets"]
        return np.frombuffer(
            data,
            dtype=dtype,
            count=(end - start) // np.dtype(dtype).itemsize,
            offset=base + start,
        ).reshape(header[name]["shape"])


def _e4m3_table() -> np.ndarray:
    """E4M3FN byte -> float32, written independently of MAX's table."""
    out = np.zeros(256, dtype=np.float64)
    for byte in range(256):
        sign = -1.0 if byte & 0x80 else 1.0
        exp, man = (byte >> 3) & 0xF, byte & 0x7
        if exp == 15 and man == 7:
            out[byte] = np.nan
        elif exp == 0:
            out[byte] = sign * man * 2.0**-9
        else:
            out[byte] = sign * (1 + man / 8) * 2.0 ** (exp - 7)
    return out.astype(np.float32)


E4M3 = _e4m3_table()
# Filled before the pool forks, so workers see them without pickling.
ARRAYS: dict[str, np.ndarray] = {}
NV: RawShards
UP: RawShards
CONFIG: PretrainedConfig


def _raw(data: WeightData) -> np.ndarray:
    assert isinstance(data.data, Buffer)
    view = {1: DType.uint8, 2: DType.uint16, 4: DType.uint32}
    return np.from_dlpack(data.data.view(view[data.dtype.size_in_bytes]))


def _source(max_name: str) -> str:
    """The checkpoint tensor a MAX weight came from."""
    name = max_name.replace(".gate.gate_score.weight", ".gate.weight")
    return name if name.startswith("lm_head") else f"model.{name}"


def _qkv_rows(max_name: str) -> tuple[int, int, int]:
    layer = re.match(r"^layers\.(\d+)\.", max_name)
    assert layer is not None
    if CONFIG.hybrid_layer_pattern[int(layer[1])] == 1:
        return _SLIDING
    return _FULL


def _classify(max_name: str, dtype: str) -> str:
    kind = "scales" if max_name.endswith("weight_scale") else "codes"
    if _STACK.match(max_name):
        return f"expert_{'e8m0' if max_name.endswith('_scale') else 'codes'}"
    if ".qkv_proj." in max_name:
        attn = "full" if _qkv_rows(max_name) == _FULL else "sliding"
        return f"qkv_{attn}_fp8_{kind}"
    if dtype == "float8_e4m3fn" or max_name.endswith("_proj.weight_scale"):
        return f"dense_mlp_fp8_{kind}"
    if max_name.endswith("gate_score.weight"):
        return "router_f32"
    return f"passthrough_{dtype}"


def _deinterleave(stack: np.ndarray) -> np.ndarray:
    """``[E, N/128, K/128, 32, 4, 4]`` interleaved scales as ``[E, N, K/32]``.

    The interleave swaps the row-atom and column-granule axes, so swapping
    them back inverts it.
    """
    experts, row_granules, col_granules = stack.shape[:3]
    return np.ascontiguousarray(stack.transpose(0, 1, 4, 3, 2, 5)).reshape(
        experts, row_granules * 128, col_granules * 4
    )


def _check(names: list[str]) -> tuple[collections.Counter[str], dict[str, str]]:
    """Checks a batch of adapted weights; returns counters and digests."""
    counts: collections.Counter[str] = collections.Counter()
    digests = {}
    for name in names:
        array = ARRAYS[name]
        digests[name] = hashlib.sha256(array.data).hexdigest()
        source = _source(name)
        if stack := _STACK.match(name):
            _check_experts(stack, array, counts, digests)
        elif name.endswith("_proj.weight") and array.dtype == np.uint8:
            _check_dense(name, array, counts)
        elif name.endswith("gate_score.weight"):
            source_bf16 = NV.array(source, np.uint16)
            counts["router_tensors"] += 1
            counts["router_inexact_upcast"] += int(
                np.count_nonzero((array >> 16) != source_bf16)
                + np.count_nonzero(array & 0xFFFF)
            )
    return counts, digests


def _check_experts(
    stack: re.Match[str],
    array: np.ndarray,
    counts: collections.Counter[str],
    digests: dict[str, str],
) -> None:
    """Compares each expert's slice of a stacked tensor with upstream."""
    layer, kind, is_scale = int(stack[1]), stack[2], bool(stack[3])
    inter = CONFIG.moe_intermediate_size
    suffix = ".weight_scale" if is_scale else ".weight"
    for expert in range(array.shape[0]):
        base = f"model.layers.{layer}.mlp.experts.{expert}"
        slices = (
            [
                ("gate_proj", array[expert, :inter]),
                ("up_proj", array[expert, inter:]),
            ]
            if kind == "gate_up"
            else [("down_proj", array[expert])]
        )
        for proj, got in slices:
            source = f"{base}.{proj}{suffix}"
            counts["expert_bytes_differ_from_upstream"] += int(
                np.count_nonzero(got != UP.array(source, np.uint8))
            )
            if is_scale:
                counts["expert_mxfp4_blocks"] += got.size
                nvfp4 = NV.array(source, np.uint8)
                counts["expert_subnormal_e4m3_scale_bytes"] += int(
                    np.count_nonzero(((nvfp4 & 0x78) == 0) & ((nvfp4 & 7) != 0))
                )
                digests[source] = hashlib.sha256(got.data).hexdigest()
            else:
                counts["expert_code_bytes"] += got.size
                counts["expert_projections"] += 1


def _check_dense(
    name: str, codes: np.ndarray, counts: collections.Counter[str]
) -> None:
    source = _source(name)
    base = name.removesuffix(".weight")
    scales = ARRAYS[f"{base}.weight_scale"].view(np.float32)
    up_codes = UP.array(source, np.uint8)
    up_scales = UP.array(
        source.removesuffix(".weight") + ".weight_scale_inv", np.float32
    )
    rows, cols = codes.shape
    counts["dense_tensors"] += 1
    counts["dense_fp8_blocks"] += scales.size
    if ".qkv_proj." in name:
        q, k, v = _qkv_rows(name)
        per_chunk = q + k + v
        chunks = codes.reshape(_CHUNKS, rows // _CHUNKS, cols)
        counts["dense_code_bytes_differ_from_upstream"] += int(
            np.count_nonzero(
                chunks[:, :per_chunk]
                != up_codes.reshape(_CHUNKS, per_chunk, cols)
            )
        )
        counts["dense_nonzero_pad_codes"] += int(
            np.count_nonzero(chunks[:, per_chunk:])
        )
    else:
        counts["dense_code_bytes_differ_from_upstream"] += int(
            np.count_nonzero(codes != up_codes)
        )
    counts["dense_scales_differ_from_upstream"] += int(
        np.count_nonzero(scales.view(np.uint32) != up_scales.view(np.uint32))
    )
    grid = np.repeat(np.repeat(scales, 128, axis=0), 128, axis=1)
    dequantized = E4M3[codes] * grid
    if ".qkv_proj." in name:
        q, k, v = _qkv_rows(name)
        chunked = dequantized.reshape(_CHUNKS, rows // _CHUNKS, cols)
        dequantized = np.concatenate(
            [
                chunked[:, :q].reshape(-1, cols),
                chunked[:, q : q + k].reshape(-1, cols),
                chunked[:, q + k : q + k + v].reshape(-1, cols),
            ]
        )
    stored = NV.array(source, np.float32)
    counts["dense_elements"] += stored.size
    counts["dense_dequant_differs_from_f32_bits"] += int(
        np.count_nonzero(dequantized.view(np.uint32) != stored.view(np.uint32))
    )


def _expect_raise(label: str, fn: Any, pattern: str) -> dict[str, str]:
    try:
        fn()
    except ValueError as error:
        message = str(error)
        if re.search(pattern, message) is None:
            raise AssertionError(f"{label}: wrong error: {message}") from error
        print(f"defect {label}: raised: {message[:160]}", flush=True)
        return {"raised": message}
    raise AssertionError(f"{label}: did not raise")


def _defects(
    state_dict: dict[str, Weights], config: PretrainedConfig, scratch: Path
) -> dict[str, Any]:
    """Injects each defect that must fail the load into an in-memory copy."""
    results = {}
    expert = "model.layers.5.mlp.experts.0.gate_proj"
    scale = _raw(state_dict[f"{expert}.weight_scale"].data()).copy()
    scale_2 = np.from_dlpack(
        state_dict[f"{expert}.weight_scale_2"].data().data
    ).astype(np.float32)

    bad = scale.copy()
    bad[7, 10:12] = 0x39
    results["non_power_of_two_scale"] = _expect_raise(
        "non_power_of_two_scale",
        lambda: e8m0_scales_from_nvfp4(bad, scale_2, expert),
        rf"{re.escape(expert)}: MXFP4 block \(row 7, block 5\)",
    )
    bad = scale.copy()
    bad[9, 20] ^= 0x08
    results["pair_halves_differ"] = _expect_raise(
        "pair_halves_differ",
        lambda: e8m0_scales_from_nvfp4(bad, scale_2, expert),
        rf"{re.escape(expert)}: MXFP4 block \(row 9, block 10\) .* differ",
    )
    missing = f"{expert}.weight_scale_2"
    results["deleted_weight_scale_2"] = _expect_raise(
        "deleted_weight_scale_2",
        lambda: convert_safetensor_state_dict(
            {k: v for k, v in state_dict.items() if k != missing}, config
        ),
        rf"missing .*{re.escape(missing)}",
    )

    extra = scratch / "extra.safetensors"
    header = {
        f"{expert}.input_scale": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
        "model.layers.1.mlp.shared_experts.up_proj.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
    }
    blob = json.dumps(header).encode()
    extra.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(8))
    injected = dict(SafetensorWeights([extra]).items())
    for label, tensor, pattern in (
        ("added_input_scale", f"{expert}.input_scale", r"input_scale tensor"),
        (
            "unknown_tensor",
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            r"neither read nor ignored.*shared_experts",
        ),
    ):
        results[label] = _expect_raise(
            label,
            lambda tensor=tensor: convert_safetensor_state_dict(
                {**state_dict, tensor: injected[tensor]}, config
            ),
            pattern,
        )
    interleaved = PretrainedConfig(**config.to_dict())
    interleaved.conversion_metadata = {
        **config.conversion_metadata,
        "qkv_layout": "tp4_interleaved",
    }
    results["qkv_layout_tp4_interleaved"] = _expect_raise(
        "qkv_layout_tp4_interleaved",
        lambda: convert_safetensor_state_dict(state_dict, interleaved),
        r"qkv_layout is 'tp4_interleaved'",
    )

    bias = "model.layers.7.mlp.gate.e_score_correction_bias"
    results["deleted_e_score_correction_bias"] = _expect_raise(
        "deleted_e_score_correction_bias",
        lambda: convert_safetensor_state_dict(
            {k: v for k, v in state_dict.items() if k != bias}, config
        ),
        rf"missing .*{re.escape(bias)}",
    )

    # One F32 element one ulp off, as (label, layer, sliding, global row, col,
    # chunk-order row block the error must name). Layer 5's row 13,348 is
    # chunk 2's V row 36, in chunk block 25, the block that also holds K.
    for label, layer, sliding, row, col, block in (
        ("f32_one_ulp_off_full_q", 0, False, 5000, 1234, 42),
        ("f32_one_ulp_off_sliding_v", 1, True, 14792, 77, 115),
        ("f32_one_ulp_off_full_straddle", 5, False, 13348, 9, 79),
    ):
        qkv = f"model.layers.{layer}.self_attn.qkv_proj.weight"
        weight = np.from_dlpack(state_dict[qkv].data().data).copy()
        weight.view(np.uint32)[row, col] ^= 1
        results[label] = _expect_raise(
            label,
            lambda qkv=qkv, weight=weight, sliding=sliding: (
                fp8_block_scaled_from_float32(
                    qkv_to_chunk_order(
                        weight, qkv_chunk_layout(config, sliding), qkv
                    ),
                    qkv,
                )
            ),
            rf"{re.escape(qkv)}: FP8 block \(row block {block},",
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvfp4", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()

    global NV, UP, CONFIG
    NV, UP = RawShards(args.nvfp4), RawShards(args.upstream)
    CONFIG = PretrainedConfig(
        **json.loads((args.nvfp4 / "config.json").read_text())
    )
    for sliding, rows in ((False, _FULL), (True, _SLIDING)):
        layout = qkv_chunk_layout(CONFIG, sliding)
        assert (layout.chunks, layout.q_rows, layout.k_rows, layout.v_rows) == (
            _CHUNKS,
            *rows,
        ), layout

    shards = sorted({args.nvfp4 / s for s in NV.weight_map.values()})
    state_dict = dict(SafetensorWeights(shards).items())
    result: dict[str, Any] = {
        "index_entries": len(NV.weight_map),
        "header_tensors": len(state_dict),
        "index_equals_headers": set(state_dict) == set(NV.weight_map),
    }
    with tempfile.TemporaryDirectory() as scratch:
        result["defects"] = _defects(state_dict, CONFIG, Path(scratch))

    start = time.time()
    adapted = convert_safetensor_state_dict(state_dict, CONFIG)
    result["adapt_seconds"] = round(time.time() - start, 1)
    print(
        f"adapted {len(adapted)} tensors in {result['adapt_seconds']}s",
        flush=True,
    )

    classes: dict[str, list[str]] = collections.defaultdict(list)
    for name, data in adapted.items():
        ARRAYS[name] = _raw(data)
        if _STACK.match(name) and name.endswith("_scale"):
            # Row-major, the layout of upstream and of the Rust adapter's
            # digests.
            ARRAYS[name] = _deinterleave(ARRAYS[name])
        classes[_classify(name, str(data.dtype).removeprefix("DType."))].append(
            name
        )
    result["adapted_tensors"] = len(adapted)
    result["class_sizes"] = {k: len(v) for k, v in sorted(classes.items())}

    stacked = sorted(n for n in adapted if _STACK.match(n))
    rest = sorted(n for n in adapted if not _STACK.match(n))
    batches = [[n] for n in stacked] + [
        rest[i : i + 64] for i in range(0, len(rest), 64)
    ]
    counts: collections.Counter[str] = collections.Counter()
    digests: dict[str, str] = {}
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for done, (c, d) in enumerate(pool.imap_unordered(_check, batches), 1):
            counts.update(c)
            digests.update(d)
            if done % 200 == 0:
                print(f"checked {done}/{len(batches)} batches", flush=True)
    result["checks"] = dict(sorted(counts.items()))

    result["sha256"] = {}
    for label, members in sorted(classes.items()):
        members.sort()
        combined = "".join(f"{n}:{digests[n]}\n" for n in members)
        result["sha256"][label] = {
            "tensors": len(members),
            "class_digest": hashlib.sha256(combined.encode()).hexdigest(),
            "first": {members[0]: digests[members[0]]},
        }
    args.out.write_text(json.dumps(result, indent=1))
    (args.out.with_suffix(".digests.json")).write_text(
        json.dumps(dict(sorted(digests.items())), indent=0)
    )
    print(
        json.dumps({k: result[k] for k in ("checks", "class_sizes")}, indent=1)
    )


if __name__ == "__main__":
    main()
