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
"""Tests the V4.1 stub's parameter tree against the checkpoint census.

The tree is the contract the weight fixtures load into, so it is checked
three ways: the stub's modules hold exactly the declared parameters, every
checkpoint tensor maps to exactly one tree path or to the allow-list, and the
stacked-experts fallback form has the leaf count it should.
"""

from __future__ import annotations

import re
from collections import Counter

import abi_contract
import pytest
from max import functional as F
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV41_modulev3.interface import (
    ParamSpec,
    V41Config,
    block_group,
    build_kv_params,
    kv_cache_config,
    parameter_specs,
    to_checkpoint_name,
    unmerged_parameter_specs,
)
from max.tensor import default_device
from stub import (
    DeepseekV41Stub,
    make_mesh,
)


def _config(model: str) -> V41Config:
    return V41Config.from_hf(abi_contract.hf_config(model))


def _tree(specs: list[ParamSpec]) -> dict[str, ParamSpec]:
    """The tree keyed by checkpoint name."""
    tree = {to_checkpoint_name(s.name): s for s in specs}
    assert len(tree) == len(specs), "two leaves share a checkpoint name"
    return tree


def test_census_table_covers_the_checkpoint() -> None:
    """The census table is the whole checkpoint census in pattern form.

    Its counts sum to the checkpoint's 188,245 tensor names, so a pattern
    lost at regeneration fails here, in CI, and not only in the env-gated
    literal-census cross-check at the bottom of this file.
    """
    text = (abi_contract.testdata_dir() / "census_patterns.tsv").read_text()
    assert sum(int(row.split("\t")[0]) for row in text.splitlines()) == 188_245


def test_full_tree_matches_census_both_ways() -> None:
    config = _config("full")
    tree = _tree(unmerged_parameter_specs(config, 8))
    census = abi_contract.census(config)
    mapped = {n for n in census if not abi_contract.allow_listed(n)}
    assert not (tree.keys() - mapped), sorted(tree.keys() - mapped)[:5]
    assert not (mapped - tree.keys()), sorted(mapped - tree.keys())[:5]
    assert not any(abi_contract.allow_listed(n) for n in tree)
    for name, spec in tree.items():
        assert census[name] == (spec.dtype, spec.shape), name


def test_full_tree_counts() -> None:
    config = _config("full")
    specs = parameter_specs(config, 8)
    experts = [s for s in specs if ".ffn.experts." in s.name]
    assert len(specs) == 185_454
    assert len(experts) == 184_320
    assert len(unmerged_parameter_specs(config, 8)) == 185_574
    assert len(parameter_specs(config, 8, "stacked")) == 1_614
    counts = Counter(
        re.sub(r"\.\d+\.", ".{L}.", s.name, count=1) for s in specs
    )
    assert counts["layers.{L}.attn.attn_sink"] == 40
    assert counts["layers.{L}.attn.compressor.wgate.weight"] == 3
    # The stacked wq_a + wkv pair: one leaf pair per layer, parts folded.
    assert counts["layers.{L}.attn.wqa_wkv.weight"] == 40
    assert counts["layers.{L}.attn.wqa_wkv.scale"] == 40
    # The stacked shared w1 + w3 pair, likewise.
    assert counts["layers.{L}.ffn.shared_experts.w13.weight"] == 40
    assert counts["layers.{L}.ffn.shared_experts.w13.scale"] == 40
    # The derived uniform NVFP4 dispatch scale: one per MoE layer.
    assert counts["layers.{L}.ffn.global_input_scale"] == 40


def test_f1_tree_counts() -> None:
    f1 = _config("f1")
    assert len(parameter_specs(f1, 2)) == 2_668
    assert len(unmerged_parameter_specs(f1, 2)) == 2_704
    without_vision = V41Config.from_hf(
        {
            k: v
            for k, v in abi_contract.hf_config("f1").items()
            if k != "vision_config"
        }
    )
    assert len(parameter_specs(without_vision, 2)) == 2_656
    # The fixture config keeps the full checkpoint's vision_config, so the
    # tree carries the vision-only router bias the fixture checkpoint also
    # carries (the registry loads it under strict_load).
    assert "layers.0.ffn.gate.bias_vl" in {
        s.name for s in parameter_specs(f1, 2)
    }
    assert "layers.0.ffn.gate.bias_vl" not in {
        s.name for s in parameter_specs(without_vision, 2)
    }


@pytest.mark.parametrize("model", ["f1", "full"])
def test_stacked_leaves_fold_their_parts(model: str) -> None:
    """Each stacked leaf is exactly its parts: the summed leading dim, the
    parts' placement, and no part left over in the folded tree."""
    from max.pipelines.architectures.deepseekV41_modulev3.interface import (
        STACKED_PROJECTIONS,
    )

    config = _config(model)
    specs = {s.name: s for s in parameter_specs(config, 2)}
    unmerged = {s.name: s for s in unmerged_parameter_specs(config, 2)}
    for layer in range(config.num_hidden_layers):
        for first, second, stacked, _ in STACKED_PROJECTIONS:
            for leaf in ("weight", "scale"):
                name = f"layers.{layer}{stacked}.{leaf}"
                parts = [
                    unmerged[f"layers.{layer}{first}.{leaf}"],
                    unmerged[f"layers.{layer}{second}.{leaf}"],
                ]
                spec = specs[name]
                assert spec.shape == (
                    parts[0].shape[0] + parts[1].shape[0],
                    *parts[0].shape[1:],
                )
                assert spec.dtype == parts[0].dtype == parts[1].dtype
                assert spec.placement == parts[0].placement
                assert f"layers.{layer}{first}.{leaf}" not in specs
                assert f"layers.{layer}{second}.{leaf}" not in specs


@pytest.mark.parametrize(("model", "tp"), [("f1", 2), ("full", 8)])
def test_stub_modules_hold_the_declared_parameters(model: str, tp: int) -> None:
    config = _config(model)
    with abi_contract.virtual_devices(tp):
        device_refs = [DeviceRef.GPU(i) for i in range(tp)]
        kv_params = build_kv_params(config, kv_cache_config(), device_refs)
        mesh = make_mesh(tp)
        with F.lazy(), default_device(mesh):
            stub = DeepseekV41Stub(config, kv_params, mesh)
        held = {
            name: (tensor.dtype, tuple(int(d) for d in tensor.shape))
            for name, tensor in stub.parameters
        }
    declared = {s.name: (s.dtype, s.shape) for s in parameter_specs(config, tp)}
    assert held.keys() == declared.keys()
    mismatched = [n for n in declared if held[n] != declared[n]]
    assert not mismatched, mismatched[:5]


def _groups(config: V41Config) -> dict[str, tuple[int, ...]]:
    by_group: dict[str, list[int]] = {}
    for layer in range(config.num_hidden_layers):
        by_group.setdefault(block_group(config, layer), []).append(layer)
    return {group: tuple(layers) for group, layers in by_group.items()}


def test_block_groups_follow_the_design() -> None:
    assert _groups(_config("full")) == {
        "swa_only": (0, 1),
        "full_r2": (2, 8, 14),
        "reuse_r2": (3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19),
        "full_r1_cand": (20,),
        "reuse_r1": (
            21,
            22,
            23,
            25,
            26,
            27,
            29,
            30,
            31,
            33,
            34,
            35,
            37,
            38,
            39,
        ),
        "reindex_r1": (24, 28, 32, 36),
    }
    assert _groups(_config("f1")) == {
        "swa_only": (0, 1),
        "full_r2": (2, 5),
        "reuse_r2": (3, 4, 6),
        "full_r1_cand": (7,),
        "reuse_r1": (8, 10),
        "reindex_r1": (9, 11),
    }
