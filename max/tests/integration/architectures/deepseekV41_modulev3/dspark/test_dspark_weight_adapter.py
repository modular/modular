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
"""CPU tests of the DSpark weight adapter.

The checkpoint's ``mtp.*`` half mapped onto the drafter's parameter tree.
The audit discipline is the strict load: every converted tensor must be a
leaf the module declares, every declared leaf must be fed, and every
checkpoint tensor is either consumed or one of the two counted drops --
``gate.bias_vl`` (the drafter routes text only) and the confidence head's
projection (dropped by default because the reference never runs adaptive
verification; ``load_confidence_head`` flips the drop into a loud load).
The dequantized
projections and the packed experts are checked bit-exactly against a numpy
decode of the same bytes, and the ModuleV3 draft's binding -- the
adapter's output as the drafter's own tree -- is checked from a real
safetensors checkpoint, through ``load_weights`` and the drafter's
``load_state_dict``.
"""

from __future__ import annotations

import dataclasses
import json
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from dspark_test_support import TESTDATA
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData, load_weights
from max.nn.legacy.kv_cache import MultiKVCacheParams
from max.pipelines.architectures.deepseekV41.weight_adapters import (
    e4m3_to_float32,
    e8m0_to_float32,
)
from max.pipelines.architectures.deepseekV41_modulev3.dspark_drafter import (
    DSparkDrafter,
)
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    PAGE_SIZE,
    V41Config,
    build_kv_params,
)
from max.pipelines.architectures.deepseekV41_modulev3.layers.rope import (
    RopeParams,
)
from max.pipelines.architectures.deepseekV41_modulev3.model_config import (
    DeepseekV41Config,
)
from max.pipelines.architectures.deepseekV41_modulev3.weight_adapters import (
    DSPARK_MODULE_PREFIX,
    as_uint8,
    convert_dspark_state_dict,
    rewire_dspark_state_dict,
    with_dtype,
)
from max.pipelines.architectures.dspark_deepseekV41_modulev3.model import (
    DSparkDeepseekV41ModuleV3DraftArchConfig,
)
from max.pipelines.lib import KVCacheConfig
from max.tensor.sharding import DeviceMesh

HIDDEN = 128
HEADS = 2
HEAD_DIM = 64
Q_LORA = 64
O_GROUPS = 2
O_LORA = 32
INTER = 64
EXPERTS = 8
VOCAB = 97
MARKOV = 16
STAGES = 3
HCMULT = 4
MIX = (2 + HCMULT) * HCMULT

_MESH = DeviceMesh((CPU(),), (1,), ("tp",))


def _full_kv_params(text: SimpleNamespace) -> MultiKVCacheParams:
    """The real KV-leaf tree at the fixture's own extents.

    The weight paths under test never touch the leaves; the tree is still
    the real one so the config carries what the serving model would.
    """
    return build_kv_params(
        V41Config.from_hf({"text_config": vars(text)}),
        KVCacheConfig(kv_cache_page_size=PAGE_SIZE),
        [DeviceRef.CPU()],
    )


def _small_config() -> DeepseekV41Config:
    """The drafter's config at this module's widths (test_dspark's shrunk
    geometry: every MXFP8 input width a multiple of 32, the noise token
    inside the shrunk vocabulary)."""
    text = SimpleNamespace(
        **json.load(open(TESTDATA / "config_full.json"))["text_config"]
    )
    kv_params = _full_kv_params(text)
    return dataclasses.replace(
        DeepseekV41Config(
            dtype=DType.bfloat16,
            kv_params=kv_params,
            devices=[DeviceRef.CPU()],
            max_seq_len=64,
            **DeepseekV41Config.resolve_text_fields(text),
        ),
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        q_lora_rank=Q_LORA,
        o_groups=O_GROUPS,
        o_lora_rank=O_LORA,
        moe_intermediate_size=INTER,
        sliding_window=64,
        dspark_n_routed_experts=EXPERTS,
        dspark_num_experts_per_tok=2,
        dspark_markov_rank=MARKOV,
        dspark_noise_token_id=96,
    )


def _rope() -> RopeParams:
    text = SimpleNamespace(
        **json.load(open(TESTDATA / "config_full.json"))["text_config"]
    )
    return RopeParams.from_text_config(text)


def _drafter(load_confidence_head: bool = False) -> DSparkDrafter:
    return DSparkDrafter(
        _small_config(),
        _MESH,
        rope=_rope(),
        load_confidence_head=load_confidence_head,
    )


def _tensor(name: str, dtype: DType, shape: tuple[int, ...]) -> WeightData:
    """Placeholder bytes at the checkpoint's stored dtype and shape."""
    if dtype == DType.bfloat16:
        return with_dtype(np.full(shape, 0x3F80, np.uint16), dtype, name)
    if dtype == DType.float32:
        return with_dtype(np.ones(shape, np.float32), dtype, name)
    if dtype == DType.int8:
        return with_dtype(np.zeros(shape, np.int8), dtype, name)
    # float8_e4m3fn values (0x38 == 1.0) and float8_e8m0fnu scales (127).
    fill = 0x38 if dtype == DType.float8_e4m3fn else 127
    return with_dtype(np.full(shape, fill, np.uint8), dtype, name)


def _fp8_linear(base: str, out_dim: int, in_dim: int) -> dict[str, WeightData]:
    """An MXFP8 linear as stored: e4m3 weight, 32x32-block scales."""
    return {
        f"{base}.weight": _tensor(
            f"{base}.weight", DType.float8_e4m3fn, (out_dim, in_dim)
        ),
        f"{base}.scale": _tensor(
            f"{base}.scale", DType.float8_e8m0fnu, (out_dim // 32, in_dim // 32)
        ),
    }


def mtp_checkpoint() -> dict[str, WeightData]:
    """The checkpoint's ``mtp.*`` tensors at this module's widths.

    Names, dtypes, and shapes follow the checkpoint census
    (``census_patterns.tsv``): the attention projections are MXFP8 with
    compact e8m0 scales, the routed experts are MXFP4 (uint8 pairs, one
    scale per row per 32 columns), and the shared expert is MXFP8.
    """
    sd: dict[str, WeightData] = {}

    def add(name: str, dtype: DType, shape: tuple[int, ...]) -> None:
        sd[name] = _tensor(name, dtype, shape)

    for stage in range(STAGES):
        p = f"mtp.{stage}"
        add(f"{p}.attn.attn_sink", DType.float32, (HEADS,))
        add(f"{p}.attn.q_norm.weight", DType.bfloat16, (Q_LORA,))
        add(f"{p}.attn.kv_norm.weight", DType.bfloat16, (HEAD_DIM,))
        sd |= _fp8_linear(f"{p}.attn.wq_a", Q_LORA, HIDDEN)
        sd |= _fp8_linear(f"{p}.attn.wq_b", HEADS * HEAD_DIM, Q_LORA)
        sd |= _fp8_linear(f"{p}.attn.wkv", HEAD_DIM, HIDDEN)
        sd |= _fp8_linear(
            f"{p}.attn.wo_a", O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS
        )
        sd |= _fp8_linear(f"{p}.attn.wo_b", HIDDEN, O_GROUPS * O_LORA)
        add(f"{p}.attn_norm.weight", DType.bfloat16, (HIDDEN,))
        add(f"{p}.ffn_norm.weight", DType.bfloat16, (HIDDEN,))
        for expert in range(EXPERTS):
            x = f"{p}.ffn.experts.{expert}"
            for proj, n, k in (
                ("w1", INTER, HIDDEN),
                ("w3", INTER, HIDDEN),
                ("w2", HIDDEN, INTER),
            ):
                add(f"{x}.{proj}.weight", DType.int8, (n, k // 2))
                add(f"{x}.{proj}.scale", DType.float8_e8m0fnu, (n, k // 32))
        add(f"{p}.ffn.gate.bias", DType.float32, (EXPERTS,))
        add(f"{p}.ffn.gate.bias_vl", DType.float32, (EXPERTS,))
        add(f"{p}.ffn.gate.weight", DType.bfloat16, (EXPERTS, HIDDEN))
        for proj, n, k in (
            ("w1", INTER, HIDDEN),
            ("w3", INTER, HIDDEN),
            ("w2", HIDDEN, INTER),
        ):
            sd |= _fp8_linear(f"{p}.ffn.shared_experts.{proj}", n, k)
        for site in ("attn", "ffn"):
            add(f"{p}.hc_{site}_base", DType.float32, (MIX,))
            add(f"{p}.hc_{site}_fn", DType.float32, (MIX, HCMULT * HIDDEN))
            add(f"{p}.hc_{site}_scale", DType.float32, (3,))
    add("mtp.0.main_norm.weight", DType.bfloat16, (HIDDEN,))
    sd |= _fp8_linear("mtp.0.main_proj", HIDDEN, HIDDEN * STAGES)
    last = f"mtp.{STAGES - 1}"
    add(f"{last}.norm.weight", DType.bfloat16, (HIDDEN,))
    add(f"{last}.markov_head.embed.weight", DType.bfloat16, (VOCAB, MARKOV))
    add(f"{last}.markov_head.head.weight", DType.bfloat16, (VOCAB, MARKOV))
    add(
        f"{last}.confidence_head.proj.weight",
        DType.bfloat16,
        (1, HIDDEN + MARKOV),
    )
    return sd


def _convert(
    sd: dict[str, WeightData], *, load_confidence_head: bool = False
) -> dict[str, WeightData]:
    return rewire_dspark_state_dict(
        sd, last_stage=STAGES - 1, load_confidence_head=load_confidence_head
    )


def _drafter_part(
    converted: dict[str, WeightData],
) -> dict[str, WeightData]:
    """The converted tensors as the drafter's own (unprefixed) state."""
    return {
        name.removeprefix(f"{DSPARK_MODULE_PREFIX}."): data
        for name, data in converted.items()
    }


def _buffer(data: WeightData) -> Buffer:
    return (
        data.data
        if isinstance(data.data, Buffer)
        else Buffer.from_dlpack(data.data)
    )


def _as_bfloat16_f32(data: WeightData) -> np.ndarray:
    """A bf16 ``WeightData``'s values, widened to float32."""
    bits = np.from_dlpack(_buffer(data).view(DType.uint16)).astype(np.uint32)
    return (bits << 16).view(np.float32)


def _dequant_oracle(weight: WeightData, scale: WeightData) -> np.ndarray:
    """``e4m3 * expanded e8m0`` in float32 -- the decode the adapter must
    reproduce, computed from the raw bytes."""
    values = e4m3_to_float32(as_uint8(weight.data))
    blocks = e8m0_to_float32(as_uint8(scale.data))
    return values * np.repeat(np.repeat(blocks, 32, axis=0), 32, axis=1)


# --------------------------------------------------------------------------- #
# The strict load: the conversion's output is exactly the drafter's tree
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("load_confidence_head", [False, True])
def test_dspark_tensors_strict_load_into_the_drafter(
    load_confidence_head: bool,
) -> None:
    """Every converted tensor is a declared leaf, every leaf is fed."""
    converted = _convert(
        mtp_checkpoint(), load_confidence_head=load_confidence_head
    )
    drafter = _drafter(load_confidence_head=load_confidence_head)
    part = _drafter_part(converted)
    declared = dict(drafter.parameters)
    assert set(declared) == set(part)
    # ``load_state_dict`` would raise on a dtype mismatch (auto_cast is
    # off), so pinning per leaf keeps a future rename loud at the leaf,
    # not at the first mismatched tensor.
    for name, weight in declared.items():
        data = part[name]
        assert data.dtype == weight.dtype, name
        assert tuple(int(d) for d in data.shape) == tuple(
            int(d) for d in weight.shape
        ), name
    drafter.load_state_dict(
        {name: data.data for name, data in part.items()}, strict=True
    )


def test_conversion_holds_no_checkpoint_names_or_dropped_tensors() -> None:
    converted = _convert(mtp_checkpoint())
    assert not any(name.startswith("mtp.") for name in converted)
    assert not any("bias_vl" in name for name in converted)
    # The default drops the confidence head's projection along with the
    # gate's image-token bias.
    assert not any("confidence_head" in name for name in converted)
    loaded = _convert(mtp_checkpoint(), load_confidence_head=True)
    assert f"{DSPARK_MODULE_PREFIX}.confidence_head.proj.weight" in loaded


def test_dropped_tensors_are_tolerant_of_absence() -> None:
    """The two drops need no checkpoint tensor: their absence loads."""
    sd = mtp_checkpoint()
    del sd[f"mtp.{STAGES - 1}.confidence_head.proj.weight"]
    for stage in range(STAGES):
        del sd[f"mtp.{stage}.ffn.gate.bias_vl"]
    converted = _convert(sd)
    drafter = _drafter()
    assert set(dict(drafter.parameters)) == set(_drafter_part(converted))


# --------------------------------------------------------------------------- #
# The draft's binding: the adapter's output as the drafter's own tree
# --------------------------------------------------------------------------- #

_SAFETENSORS_DTYPES = {
    DType.bfloat16: "BF16",
    DType.float32: "F32",
    DType.int8: "I8",
    DType.float8_e4m3fn: "F8_E4M3",
    DType.float8_e8m0fnu: "F8_E8M0",
}
"""The header spellings of the fixture's stored dtypes."""

_RAW_VIEWS = {
    DType.bfloat16: DType.uint16,
    DType.float32: DType.uint32,
    DType.int8: DType.uint8,
    DType.float8_e4m3fn: DType.uint8,
    DType.float8_e8m0fnu: DType.uint8,
}
"""Same-width unsigned views, so the writer copies bytes verbatim."""


def _write_safetensors(path: Path, sd: dict[str, WeightData]) -> None:
    """Writes ``sd`` as one real safetensors checkpoint, bytes verbatim.

    The header spellings are the ABI suite's own, so ``load_weights``
    reads each tensor back at its stored dtype -- the lazy leaves a real
    checkpoint hands the adapter, never a hand-built mapping.
    """
    header: dict[str, Any] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, data in sd.items():
        raw = np.from_dlpack(_buffer(data).view(_RAW_VIEWS[data.dtype]))
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[data.dtype],
            "shape": [int(dim) for dim in data.shape],
            "data_offsets": [offset, offset + raw.nbytes],
        }
        blobs.append(raw.tobytes())
        offset += raw.nbytes
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as file:
        file.write(struct.pack("<Q", len(blob)))
        file.write(blob)
        for chunk in blobs:
            file.write(chunk)


def test_the_draft_checkpoint_weights_bind_the_adapter_s_output(
    tmp_path: Path,
) -> None:
    """The adapter's ``dspark.*`` names load into the drafter as its own.

    The real chain end to end, nothing renamed by hand: the checkpoint
    fixture written as one real safetensors file, read back through
    ``load_weights`` -- the lazy leaves the pairing hands the draft --
    converted by the ModuleV3 draft's ``checkpoint_weights``, which the
    pairing keys under ``draft.``, and loaded through the drafter's own
    ``load_state_dict``. A name that kept the adapter's prefix would
    leave the drafter's own leaves unfed, which the load refuses.
    """
    checkpoint = tmp_path / "dspark.safetensors"
    _write_safetensors(checkpoint, mtp_checkpoint())
    lazy = load_weights([checkpoint])
    converted = convert_dspark_state_dict(dict(lazy.items()), num_stages=STAGES)
    weights = DSparkDeepseekV41ModuleV3DraftArchConfig.checkpoint_weights(
        cast(Any, SimpleNamespace(weights=lazy)), _small_config()
    )
    assert "main_proj.weight" in weights
    assert not any(name.startswith("dspark.") for name in weights)
    drafter = _drafter()
    drafter.load_state_dict(weights)

    d = DSPARK_MODULE_PREFIX
    # The hoisted projection keeps the checkpoint's MXFP8 pair: the
    # loaded leaf holds the adapter's e4m3 bytes, byte for byte.
    got = torch.from_dlpack(drafter.main_proj.weight.to(CPU()))
    np.testing.assert_array_equal(
        got.view(torch.uint8).numpy(),
        as_uint8(converted[f"{d}.main_proj.weight"].data),
    )
    # A stage's dequantized projection, and its fp32 sink.
    params = dict(drafter.parameters)
    got = torch.from_dlpack(params["stages.1.attn.wq_b.weight"].to(CPU()))
    np.testing.assert_array_equal(
        got.to(torch.float32).numpy(),
        _as_bfloat16_f32(converted[f"{d}.stages.1.attn.wq_b.weight"]),
    )
    np.testing.assert_array_equal(
        params["stages.0.attn.attn_sink"].to_numpy(),
        np.from_dlpack(_buffer(converted[f"{d}.stages.0.attn.attn_sink"])),
    )


# --------------------------------------------------------------------------- #
# The transforms, bit-exact against a numpy decode of the same bytes
# --------------------------------------------------------------------------- #


def test_dequantized_projections_match_the_decode_oracle() -> None:
    """The bf16 projections are exactly ``e4m3 * expanded scale``."""
    sd = mtp_checkpoint()
    rng = np.random.default_rng(0)
    # Distinct, finite e4m3 codes (excluding the 0x7F/0xFF NaNs) and
    # scales, so every dequantized value carries a real scale factor.
    for base in ("mtp.1.attn.wq_a", "mtp.1.attn.wo_a"):
        shape = tuple(int(d) for d in sd[f"{base}.weight"].shape)
        sd[f"{base}.weight"] = with_dtype(
            rng.integers(0, 0x7E, size=shape, dtype=np.uint8),
            DType.float8_e4m3fn,
            f"{base}.weight",
        )
        sshape = tuple(int(d) for d in sd[f"{base}.scale"].shape)
        sd[f"{base}.scale"] = with_dtype(
            rng.integers(120, 130, size=sshape, dtype=np.uint8),
            DType.float8_e8m0fnu,
            f"{base}.scale",
        )
    converted = _convert(sd)
    for proj in ("wq_a", "wo_a"):
        got = _as_bfloat16_f32(
            converted[f"{DSPARK_MODULE_PREFIX}.stages.1.attn.{proj}.weight"]
        )
        expected = _dequant_oracle(
            sd[f"mtp.1.attn.{proj}.weight"], sd[f"mtp.1.attn.{proj}.scale"]
        )
        np.testing.assert_array_equal(got, expected)


def test_expert_stacking_keeps_expert_order_and_gate_rows_first() -> None:
    """``gate_up_proj`` stacks expert-major with ``w1``'s rows first."""
    sd = mtp_checkpoint()
    rng = np.random.default_rng(1)
    w1 = rng.integers(0, 200, (EXPERTS, INTER, HIDDEN // 2), dtype=np.uint8)
    w3 = rng.integers(0, 200, (EXPERTS, INTER, HIDDEN // 2), dtype=np.uint8)
    w2 = rng.integers(0, 200, (EXPERTS, HIDDEN, INTER // 2), dtype=np.uint8)
    s1 = rng.integers(120, 130, (EXPERTS, INTER, HIDDEN // 32), dtype=np.uint8)
    s2 = rng.integers(120, 130, (EXPERTS, HIDDEN, INTER // 32), dtype=np.uint8)
    for expert in range(EXPERTS):
        x = f"mtp.0.ffn.experts.{expert}"
        sd[f"{x}.w1.weight"] = with_dtype(
            w1[expert], DType.int8, f"{x}.w1.weight"
        )
        sd[f"{x}.w3.weight"] = with_dtype(
            w3[expert], DType.int8, f"{x}.w3.weight"
        )
        sd[f"{x}.w2.weight"] = with_dtype(
            w2[expert], DType.int8, f"{x}.w2.weight"
        )
        sd[f"{x}.w1.scale"] = with_dtype(
            s1[expert], DType.float8_e8m0fnu, f"{x}.w1.scale"
        )
        sd[f"{x}.w3.scale"] = with_dtype(
            s1[expert] + 1, DType.float8_e8m0fnu, f"{x}.w3.scale"
        )
        sd[f"{x}.w2.scale"] = with_dtype(
            s2[expert], DType.float8_e8m0fnu, f"{x}.w2.scale"
        )
    converted = _convert(sd)
    base = f"{DSPARK_MODULE_PREFIX}.stages.0.ffn.experts"
    gate_up = as_uint8(converted[f"{base}.gate_up_proj.weight"].data)
    gate_up_scale = as_uint8(
        converted[f"{base}.gate_up_proj.weight_scale"].data
    )
    down = as_uint8(converted[f"{base}.down_proj.weight"].data)
    down_scale = as_uint8(converted[f"{base}.down_proj.weight_scale"].data)
    assert gate_up.shape == (EXPERTS, 2 * INTER, HIDDEN // 2)
    np.testing.assert_array_equal(gate_up[:, :INTER], w1)
    np.testing.assert_array_equal(gate_up[:, INTER:], w3)
    np.testing.assert_array_equal(gate_up_scale[:, :INTER], s1)
    np.testing.assert_array_equal(gate_up_scale[:, INTER:], s1 + 1)
    np.testing.assert_array_equal(down, w2)
    np.testing.assert_array_equal(down_scale, s2)


def test_hoisted_tensors_come_from_the_owning_stages() -> None:
    """``main_*`` hoists from stage 0; ``norm``/heads from the last."""
    sd = mtp_checkpoint()
    sd["mtp.0.main_norm.weight"] = with_dtype(
        np.full((HIDDEN,), 0x3F00, np.uint16),
        DType.bfloat16,
        "mtp.0.main_norm.weight",
    )
    sd[f"mtp.{STAGES - 1}.norm.weight"] = with_dtype(
        np.full((HIDDEN,), 0x3F80, np.uint16),
        DType.bfloat16,
        f"mtp.{STAGES - 1}.norm.weight",
    )
    converted = _convert(sd)
    d = DSPARK_MODULE_PREFIX
    assert converted[f"{d}.main_norm.weight"].data is not None
    got_main = _as_bfloat16_f32(converted[f"{d}.main_norm.weight"])
    got_norm = _as_bfloat16_f32(converted[f"{d}.norm.weight"])
    np.testing.assert_array_equal(got_main, np.full(HIDDEN, 0.5, np.float32))
    np.testing.assert_array_equal(got_norm, np.ones(HIDDEN, np.float32))


def test_confidence_head_widens_bfloat16_to_float32() -> None:
    sd = mtp_checkpoint()
    key = f"mtp.{STAGES - 1}.confidence_head.proj.weight"
    sd[key] = with_dtype(
        np.full((1, HIDDEN + MARKOV), 0x3F80, np.uint16),
        DType.bfloat16,
        key,
    )
    converted = _convert(sd, load_confidence_head=True)
    loaded = converted[f"{DSPARK_MODULE_PREFIX}.confidence_head.proj.weight"]
    assert loaded.dtype == DType.float32
    np.testing.assert_array_equal(
        np.from_dlpack(_buffer(loaded)).reshape(-1)[:3],
        np.ones(3, np.float32),
    )


# --------------------------------------------------------------------------- #
# Fail-loud: nothing skips, nothing hides
# --------------------------------------------------------------------------- #


def test_a_requested_confidence_head_missing_fails_loud() -> None:
    sd = mtp_checkpoint()
    del sd[f"mtp.{STAGES - 1}.confidence_head.proj.weight"]
    with pytest.raises(ValueError, match=r"confidence_head\.proj\.weight"):
        _convert(sd, load_confidence_head=True)


def test_an_unknown_mtp_tensor_fails_loud() -> None:
    """A name outside the census's patterns never slips through."""
    sd = mtp_checkpoint()
    sd["mtp.0.attn.bogus.weight"] = _tensor(
        "mtp.0.attn.bogus.weight", DType.bfloat16, (4,)
    )
    with pytest.raises(ValueError, match="unconsumed DSpark tensors"):
        _convert(sd)


def test_a_hoisted_tensor_under_the_wrong_stage_fails_loud() -> None:
    sd = mtp_checkpoint()
    del sd["mtp.0.main_norm.weight"]
    sd["mtp.1.main_norm.weight"] = _tensor(
        "mtp.1.main_norm.weight", DType.bfloat16, (HIDDEN,)
    )
    with pytest.raises(ValueError, match="missing from the DSpark stages"):
        _convert(sd)


def test_a_dequant_weight_without_its_scale_fails_loud() -> None:
    sd = mtp_checkpoint()
    del sd["mtp.0.attn.wkv.scale"]
    with pytest.raises(ValueError, match=r"wkv\.scale"):
        _convert(sd)


def test_non_dense_expert_ids_fail_loud() -> None:
    sd = mtp_checkpoint()
    for name in list(sd):
        if name.startswith("mtp.0.ffn.experts.3."):
            del sd[name]
    with pytest.raises(ValueError, match="expert ids are not dense"):
        _convert(sd)


def test_a_non_mtp_tensor_fails_loud() -> None:
    with pytest.raises(ValueError, match="not a DSpark stage"):
        rewire_dspark_state_dict(
            {"norm.weight": _tensor("norm.weight", DType.bfloat16, (4,))},
            last_stage=STAGES - 1,
        )


def test_convert_dspark_state_dict_reads_the_lazy_weights() -> None:
    """The pipeline-facing entry reads ``mtp.*`` out of the mapping."""
    lazy: dict[str, MagicMock] = {}
    for name, data in mtp_checkpoint().items():
        mock = MagicMock()
        mock.data.return_value = data
        lazy[name] = mock
    lazy["embed.weight"] = MagicMock()
    converted = convert_dspark_state_dict(
        lazy,
        num_stages=STAGES,
    )
    assert set(converted) == set(_convert(mtp_checkpoint()))
    assert not lazy["embed.weight"].data.called


def test_an_empty_checkpoint_fails_loud() -> None:
    with pytest.raises(ValueError, match="no mtp"):
        convert_dspark_state_dict({}, num_stages=STAGES)
