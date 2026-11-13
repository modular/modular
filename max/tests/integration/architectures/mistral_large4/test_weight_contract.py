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
"""The Mistral Large 4 name/shape/dtype/scale contract, on both loaders.

Two gates live here:

* The committed fixture gate runs everywhere. It asserts MAX's table equals the
  golden, that every tensor in the fixture is consumed or on the documented skip
  list, and (when ``ML4_RUST_CONTRACT`` names the JSON the Rust producer
  printed) that Mach's table is the golden too.
* The real-checkpoint gate is manual: ``ML4_HEADERS`` names a headers-only JSON
  dumped from the published checkpoint and ``ML4_CHECKPOINT_DIR`` points at the
  checkpoint itself. Neither test reads tensor payloads except the per-tensor
  NVFP4 scalars in the gate/up check, which are four bytes each.
"""

from __future__ import annotations

import json
import os
import struct
from collections import Counter
from pathlib import Path
from typing import TypedDict

import pytest
from max.pipelines.architectures.mistral_large4_modulev3.model_config import (
    QUANT_ALGO_NVFP4,
    MistralLarge4NormalizedConfig,
    read_params_json,
)
from max.pipelines.architectures.mistral_large4_modulev3.weight_contract import (
    QUANT_ALGO_FP8_PB_WO,
    ROLE_FP8_SCALE,
    ROLE_NVFP4_GLOBAL_SCALE,
    ROLE_NVFP4_GROUP_SCALE,
    ROLE_NVFP4_INPUT_SCALE,
    ROLE_NVFP4_WEIGHT,
    ROLE_PLAIN,
    ROLE_VISION_QK,
    ContractTable,
    is_vision_name,
    skip_reason,
    skipped_tensor_documentation,
    transform_vision_weight_name,
    transform_weight_name,
    vision_weight_table,
    weight_table,
)

_TESTDATA = Path(__file__).parent / "testdata"


class _TensorHeader(TypedDict):
    """One entry of a shard's safetensors header."""

    dtype: str
    shape: list[int]
    data_offsets: list[int]


def _read_headers(path: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Reads a headers fixture as ``name -> (dtype, shape)``."""
    raw = json.loads(path.read_text())
    return {
        name: (entry["dtype"], tuple(entry["shape"]))
        for name, entry in raw.items()
    }


def _rows(
    table: ContractTable,
) -> dict[str, tuple[str, str, tuple[int, ...], str]]:
    """The table in the golden's own shape, keyed by checkpoint name."""
    return {
        tensor.source: (
            tensor.name,
            tensor.dtype,
            tensor.shape,
            tensor.role,
        )
        for tensor in table.tensors
    }


def _golden() -> dict[str, object]:
    return json.loads((_TESTDATA / "ml4_nvfp4_contract.json").read_text())


def _golden_rows(
    golden: dict[str, object],
) -> dict[str, tuple[str, str, tuple[int, ...], str]]:
    tensors = golden["tensors"]
    assert isinstance(tensors, list)
    return {
        entry["source"]: (
            entry["name"],
            entry["dtype"],
            tuple(entry["shape"]),
            entry["role"],
        )
        for entry in tensors
    }


def test_python_adapter_matches_the_golden_table() -> None:
    fixture = _read_headers(_TESTDATA / "ml4_nvfp4_headers.json")
    # The golden is this table verbatim, so the comparison covers the skip list
    # and its reasons as well as every name/shape/dtype/role row.
    assert weight_table(fixture).to_json() == _golden()


def test_every_fixture_tensor_is_consumed_or_skipped() -> None:
    fixture = _read_headers(_TESTDATA / "ml4_nvfp4_headers.json")
    table = weight_table(fixture)
    assert len(table.tensors) + len(table.skipped) == len(fixture)
    # Consumed exactly once: the module names are unique and account for every
    # non-skipped source name.
    assert len({tensor.name for tensor in table.tensors}) == len(table.tensors)
    assert len({tensor.source for tensor in table.tensors}) == len(
        table.tensors
    )
    skipped_sources = {source for source, _ in table.skipped}
    assert skipped_sources == {
        name for name in fixture if skip_reason(name) is not None
    }


def test_the_skip_list_names_the_drafter_and_the_vision_tower() -> None:
    fixture = _read_headers(_TESTDATA / "ml4_nvfp4_headers.json")
    table = weight_table(fixture)
    reasons = Counter(reason for _, reason in table.skipped)
    assert set(reasons) == {"dspark", "vision"}
    documented = {prefix for prefix, _, _ in skipped_tensor_documentation()}
    assert documented == {
        "mtp_dspark.",
        "vision_encoder.",
        "pre_mm_projector_norm.",
        "patch_merger.",
        "vision_language_adapter.",
    }
    for tensor in table.tensors:
        assert not any(tensor.source.startswith(p) for p in documented)


def test_the_role_census_matches_one_scale_per_quantized_weight() -> None:
    fixture = _read_headers(_TESTDATA / "ml4_nvfp4_headers.json")
    census = Counter(tensor.role for tensor in weight_table(fixture).tensors)
    assert census[ROLE_FP8_SCALE] == census["fp8_weight"]
    assert census[ROLE_NVFP4_GROUP_SCALE] == census[ROLE_NVFP4_WEIGHT]
    assert census[ROLE_NVFP4_GLOBAL_SCALE] == census[ROLE_NVFP4_WEIGHT]
    assert census[ROLE_NVFP4_INPUT_SCALE] == census[ROLE_NVFP4_WEIGHT]
    assert census[ROLE_FP8_SCALE] > 0
    assert census[ROLE_NVFP4_WEIGHT] > 0
    assert census[ROLE_PLAIN] > 0


def test_the_two_scale_conventions_do_not_share_a_role() -> None:
    """A 4-D ModelOpt block scale and an NVFP4 group scale are told apart.

    The sibling is what decides: an expert's `.weight_scale` sits beside
    `.weight_scale_2` and keeps its 2-D `float8_e4m3fn` grid, while `wq`'s does
    not and squashes onto the 2-D block grid as `weight_scale_inv`.
    """
    headers = {
        "layers.0.attention.wq.weight": ("F8_E4M3", (128, 128)),
        "layers.0.attention.wq.weight_scale": ("F32", (1, 1, 1, 1)),
        "layers.1.experts.0.w1.weight": ("U8", (128, 8)),
        "layers.1.experts.0.w1.weight_scale": ("F8_E4M3", (128, 1)),
        "layers.1.experts.0.w1.weight_scale_2": ("F32", ()),
        "layers.1.experts.0.w1.input_scale": ("F32", ()),
    }
    table = weight_table(headers)
    rows = _rows(table)
    assert (
        rows["layers.0.attention.wq.weight_scale"][0]
        == "language_model.layers.0.self_attn.wq.weight_scale_inv"
    )
    assert rows["layers.0.attention.wq.weight_scale"][2] == (1, 1)
    assert rows["layers.0.attention.wq.weight_scale"][3] == ROLE_FP8_SCALE
    assert (
        rows["layers.1.experts.0.w1.weight_scale"][0]
        == "language_model.layers.1.mlp.experts.0.w1.weight_scale"
    )
    assert rows["layers.1.experts.0.w1.weight_scale"][2] == (128, 1)
    assert (
        rows["layers.1.experts.0.w1.weight_scale"][3] == ROLE_NVFP4_GROUP_SCALE
    )
    assert (
        rows["layers.1.experts.0.w1.weight_scale_2"][3]
        == ROLE_NVFP4_GLOBAL_SCALE
    )
    assert (
        rows["layers.1.experts.0.w1.input_scale"][3] == ROLE_NVFP4_INPUT_SCALE
    )


def test_a_transposed_block_grid_is_refused() -> None:
    headers = {
        "layers.0.attention.wq.weight": ("F8_E4M3", (128, 256)),
        "layers.0.attention.wq.weight_scale": ("F32", (2, 1, 1, 1)),
    }
    with pytest.raises(ValueError, match="128x128"):
        weight_table(headers)


def test_a_scale_without_its_weight_sibling_is_refused() -> None:
    """A scale whose weight is absent names the module, not a bare KeyError."""
    headers = {"layers.0.attention.wq.weight_scale": ("F32", (1, 1, 1, 1))}
    with pytest.raises(ValueError, match=r"`\.weight` sibling"):
        weight_table(headers)


def test_the_rename_table_keeps_attention_norm_and_shared_experts_apart() -> (
    None
):
    assert (
        transform_weight_name("layers.0.attention_norm.weight")
        == "language_model.layers.0.input_layernorm.weight"
    )
    assert (
        transform_weight_name("layers.0.attention.wq.weight")
        == "language_model.layers.0.self_attn.wq.weight"
    )
    assert (
        transform_weight_name("layers.1.shared_experts.w1.weight")
        == "language_model.layers.1.mlp.shared_experts.w1.weight"
    )
    assert (
        transform_weight_name("layers.1.experts.127.w3.weight")
        == "language_model.layers.1.mlp.experts.127.w3.weight"
    )
    assert (
        transform_weight_name("output.weight")
        == "language_model.lm_head.weight"
    )


def test_two_sources_mapping_to_one_module_name_are_refused() -> None:
    """`output.weight` and a literal `lm_head.weight` land on one module name."""
    headers = {
        "output.weight": ("BF16", (8, 8)),
        "lm_head.weight": ("BF16", (8, 8)),
    }
    with pytest.raises(ValueError, match="collide"):
        weight_table(headers)


def test_a_base_fp8_scale_is_refused() -> None:
    headers = {
        "layers.0.attention.wq.weight": ("F8_E4M3", (128, 128)),
        "layers.0.attention.wq.qscale_weight": ("F32", (1, 1)),
    }
    with pytest.raises(ValueError, match="qscale_weight"):
        weight_table(headers)


def test_the_real_checkpoint_headers_adapt_with_zero_mismatches() -> None:
    """Manual gate: one real checkpoint's headers against both loaders.

    ``ML4_HEADERS`` is a headers-only JSON produced by reading the safetensors
    headers of a published or minimized ML4 checkpoint (no tensor payload), and
    ``ML4_RUST_CONTRACT`` is the JSON Mach's ``ml4_weight_contract`` producer
    printed from the same file. The committed fixture gate covers MAX against
    the golden and `ml4_contract.rs` covers Mach against it; this test is the
    live diff of the two.
    """
    headers_path = os.environ.get("ML4_HEADERS")
    if not headers_path:
        pytest.skip("set ML4_HEADERS to a real checkpoint's headers JSON")
    rust_path = os.environ.get("ML4_RUST_CONTRACT")
    if not rust_path:
        pytest.skip("set ML4_RUST_CONTRACT to Mach's table for those headers")
    fixture = _read_headers(Path(headers_path))
    table = weight_table(fixture)
    mach_rows = _golden_rows(json.loads(Path(rust_path).read_text()))
    python_rows = _rows(table)
    mismatches = [
        source
        for source in set(mach_rows) | set(python_rows)
        if mach_rows.get(source) != python_rows.get(source)
    ]
    assert not mismatches, (
        f"{len(mismatches)} of {len(fixture)} tensors mismatch, first few: "
        f"{[(source, mach_rows.get(source), python_rows.get(source)) for source in sorted(mismatches)[:3]]}"
    )
    assert len(table.tensors) + len(table.skipped) == len(fixture)


def test_the_real_checkpoint_gate_up_globals_are_equal_per_expert() -> None:
    """Manual gate: ``w1``/``w3`` NVFP4 globals must match per expert.

    ``concat_nvfp4`` takes ``weight_scale_2`` and ``input_scale`` from the first
    tensor of the fused gate||up leaf, and pre-quantized tensors are never
    re-pinned, so unequal globals silently mis-scale the up projection. This
    reads only those scalar tensors, never a weight.
    """
    directory = os.environ.get("ML4_CHECKPOINT_DIR")
    if not directory:
        pytest.skip("set ML4_CHECKPOINT_DIR to the checkpoint directory")
    root = Path(directory)
    index = json.loads(
        (root / "consolidated.safetensors.index.json").read_text()
    )
    weight_map = index["weight_map"]

    files: dict[str, tuple[Path, dict[str, _TensorHeader], int]] = {}

    def scalar(name: str) -> float:
        shard = weight_map[name]
        if shard not in files:
            path = root / shard
            with open(path, "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
            files[shard] = (path, header, 8 + length)
        path, header, data_start = files[shard]
        start, _ = header[name]["data_offsets"]
        with open(path, "rb") as handle:
            handle.seek(data_start + start)
            return struct.unpack("<f", handle.read(4))[0]

    weights = [name for name in weight_map if name.endswith(".w1.input_scale")]
    assert weights, "the checkpoint holds NVFP4 experts"
    differing_globals = []
    differing_inputs = []
    for w1 in weights:
        stem = w1[: -len(".w1.input_scale")]
        w3_scale = f"{stem}.w3.weight_scale_2"
        w1_scale = f"{stem}.w1.weight_scale_2"
        if scalar(w1_scale) != scalar(w3_scale):
            differing_globals.append(stem)
        if scalar(f"{stem}.w1.input_scale") != scalar(f"{stem}.w3.input_scale"):
            differing_inputs.append(stem)
    assert not differing_globals, (
        f"{len(differing_globals)} expert(s) disagree on the gate/up weight "
        f"global, first: {differing_globals[:3]}"
    )
    assert not differing_inputs, (
        f"{len(differing_inputs)} expert(s) disagree on the gate/up activation "
        f"global, first: {differing_inputs[:3]}"
    )


def test_the_quantized_layer_map_covers_every_expert_layer() -> None:
    """Every MoE layer's experts are NVFP4 and layer 0's MLP is FP8."""
    params = read_params_json(_TESTDATA / "ml4_nvfp4_params.json")
    config = MistralLarge4NormalizedConfig.from_params_json(params)
    verdicts = [
        config.quantization.algo_for(f"layers.{i}.experts") for i in range(72)
    ]
    assert verdicts[1:] == [QUANT_ALGO_NVFP4] * 71
    assert verdicts[0] == "BF16"
    assert config.quantization.algo_for("layers.0.feed_forward.w1") == (
        QUANT_ALGO_FP8_PB_WO
    )


def test_the_vision_table_binds_every_vision_tensor_under_its_own_names() -> (
    None
):
    """The vision graph consumes the whole vision subtree, unrenamed.

    The one exception is the conv patch embed, a raw graph tensor. The text
    table skips exactly these sources, and the text replacements would
    rewrite the tower attention names, which is why the vision route is
    identity-except-one and never goes through the text table.
    """
    fixture = _read_headers(_TESTDATA / "ml4_nvfp4_headers.json")
    vision = vision_weight_table(fixture)
    sources = {name for name in fixture if is_vision_name(name)}
    assert sources, "the fixture holds the vision subtree"
    assert {tensor.source for tensor in vision.tensors} == sources
    rows = {tensor.source: tensor for tensor in vision.tensors}
    assert len(rows) == len(vision.tensors)
    patch_conv = rows["vision_encoder.patch_conv.weight"]
    assert patch_conv.name == "vision_encoder.patch_conv"
    for source, tensor in rows.items():
        if source != "vision_encoder.patch_conv.weight":
            assert tensor.name == source
        assert tensor.dtype == "BF16"
        if source.endswith((".attention.wq.weight", ".attention.wk.weight")):
            # The tower's Q/K rows are permuted into the rotate-half rope
            # layout at load; every other vision tensor is plain.
            assert tensor.role == ROLE_VISION_QK
        else:
            assert tensor.role == ROLE_PLAIN

    text_table = weight_table(fixture)
    assert not sources & {tensor.source for tensor in text_table.tensors}
    assert {source for source, _ in text_table.skipped} >= sources
    # The text table grafts and renames; a vision name routed through it
    # would land under `language_model.` with a rewritten attention path.
    wq = rows["vision_encoder.transformer.layers.0.attention.wq.weight"]
    assert (
        transform_weight_name(wq.source)
        == "language_model.vision_encoder.transformer.layers.0.self_attn.wq.weight"
    )
    assert transform_vision_weight_name(wq.source) == wq.source
    assert wq.role == ROLE_VISION_QK
    wk = rows["vision_encoder.transformer.layers.0.attention.wk.weight"]
    assert wk.role == ROLE_VISION_QK


def test_a_quantized_vision_tensor_is_refused() -> None:
    """The vision subtree is BF16; a packed weight is a different loader."""
    headers = {
        "vision_encoder.ln_pre.weight": ("BF16", (2048,)),
        "vision_encoder.transformer.layers.0.attention.wq.weight": (
            "U8",
            (2048, 2048),
        ),
    }
    with pytest.raises(ValueError, match="unquantized BF16"):
        vision_weight_table(headers)


def test_a_mis_shaped_vision_qk_projection_is_refused() -> None:
    """The Q/K rope-layout permutation needs the tower's 2048 rope rows."""
    headers = {
        "vision_encoder.transformer.layers.0.attention.wq.weight": (
            "BF16",
            (1024, 2048),
        ),
    }
    with pytest.raises(ValueError, match="rope-layout permutation"):
        vision_weight_table(headers)
