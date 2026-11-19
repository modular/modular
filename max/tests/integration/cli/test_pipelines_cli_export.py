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
"""Tests for ``max export``."""

import hashlib
import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner
from max import driver
from max._artifact import export as export_module
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.format_spec import FormatSpec
from max._artifact.serialization import deserialize
from max._entrypoints import pipelines
from max.pipelines.lib.config import PipelineConfig


def _env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"{name} environment variable is not set")
    return value


def _export_args(output: Path) -> list[str]:
    return [
        "export",
        "--model-path",
        _env("TEST_LLAMA_REPO"),
        f"--custom-architectures={_env('PIPELINES_CUSTOM_ARCHITECTURE')}",
        "--no-prefer-module-v3",
        "--devices=cpu",
        "--quantization-encoding=float32",
        "--max-length=128",
        "--max-batch-size=1",
        "--output",
        str(output),
    ]


def test_export_writes_the_graphs_by_role(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        pipelines.main, _export_args(tmp_path / "dummy"), catch_exceptions=False
    )
    dest = tmp_path / "dummy.m"
    assert result.exit_code == 0, result.output
    assert f"Exported pipeline artifact to {dest}" in result.output

    sections = json.loads((dest / "artifact.json").read_text())
    assert "format_spec" not in sections
    # A host-only artifact cannot overlap, so it ships no overlap helpers.
    section = sections["model"]
    assert set(section) == {"decoder", "sampler"}
    assert section["decoder"]["mef"] == {"$file": "decoder/model.mef"}
    engine_config = sections["engine_config"]
    assert engine_config["max_batch_size"] == 1
    assert engine_config["max_length"] == 128
    assert engine_config["parallel"] == {
        "dp": 1,
        "tp": 1,
        "ep": 1,
        "dcp": 1,
        "pcp": 1,
        "dcp_interleave": 1,
    }
    assert engine_config["prefill_token_budget"] is None
    assert engine_config["speculative"] is None
    # Llama's one attention cache, which is not in a tree, and its pool leaf.
    caches = sections["kv_cache"]["caches"]
    assert [(cache["name"], cache["kind"]) for cache in caches] == [("", "mha")]
    assert [leaf["leaf_id"] for leaf in caches[0]["leaves"]] == ["full_group"]
    # The decoder's one entry; the sampler loads no checkpoint.
    assert sections["weight_loader"] == {
        "decoder": [
            {
                "checkpoint": "target",
                "rules": [
                    {
                        "rule": "WeightRenaming",
                        "pattern": r"^model\.",
                        "replacement": "",
                    }
                ],
            }
        ]
    }
    loaded = deserialize(dest)
    assert loaded.model is not None
    assert set(loaded.model.roles) == {"decoder", "sampler"}
    assert loaded.weight_loader is not None
    assert set(loaded.weight_loader.graphs) == {"decoder"}


def test_export_names_the_decoder_inputs_and_bounds_each_step(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        pipelines.main, _export_args(tmp_path / "named"), catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    dest = tmp_path / "named.m"
    sections = json.loads((dest / "artifact.json").read_text())

    in_tree = json.dumps(sections["model"]["decoder"]["in_tree"])
    assert '"ragged_batch"' in in_tree
    assert '"paged_kv"' in in_tree

    dims = sections["batch_processor"]["dims"]
    # The KV blocks are a buffer the cache sizes, so their pages get no bound.
    assert set(dims) == {
        "total_seq_len",
        "replica_0_batch_size",
        "replica_0_max_num_pages",
    }
    assert dims["replica_0_batch_size"] == 1
    assert deserialize(dest).batch_processor == BatchProcessorDims(dims)


def test_export_stores_the_format_spec_it_is_given(tmp_path: Path) -> None:
    document = tmp_path / "spec.json"
    document.write_text('{"format": "formatspec.spec", "spec_version": 1}\n')
    result = CliRunner().invoke(
        pipelines.main,
        [*_export_args(tmp_path / "spec"), "--format-spec", str(document)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    dest = tmp_path / "spec.m"

    sections = json.loads((dest / "artifact.json").read_text())
    assert sections["format_spec"] == {
        "document": {"$file": "format_spec.json"}
    }
    stored = (dest / "format_spec.json").read_bytes()
    assert stored == document.read_bytes()
    metadata = json.loads((dest / "metadata.json").read_text())
    assert (
        metadata["files"]["format_spec.json"]
        == hashlib.sha256(stored).hexdigest()
    )
    assert deserialize(dest).format_spec == FormatSpec(
        dest / "format_spec.json"
    )


def test_export_target_configures_virtual_devices_before_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[tuple[str, object]] = []
    set_api = driver.set_virtual_device_api
    set_arch = driver.set_virtual_device_target_arch
    set_count = driver.set_virtual_device_count

    def api(value: str) -> None:
        events.append(("api", value))
        set_api(value)

    def arch(value: str) -> None:
        events.append(("arch", value))
        set_arch(value)

    def count(value: int) -> None:
        events.append(("count", value))
        set_count(value)

    monkeypatch.setattr(driver, "set_virtual_device_api", api)
    monkeypatch.setattr(driver, "set_virtual_device_target_arch", arch)
    monkeypatch.setattr(driver, "set_virtual_device_count", count)

    def export(
        config: PipelineConfig,
        output: Path,
        format_spec: Path | None,
        *,
        prefill_token_budget: int | None,
        prefill_capture_tokens: int,
    ) -> Path:
        assert driver.is_virtual_device_mode()
        assert len(config.model.device_specs) == 2
        events.append(
            ("export", (prefill_token_budget, prefill_capture_tokens))
        )
        return output.with_suffix(".m")

    monkeypatch.setattr(export_module, "export_pipeline", export)
    args = [
        "--devices=gpu:0,1" if arg == "--devices=cpu" else arg
        for arg in _export_args(tmp_path / "virtual")
    ]
    try:
        result = CliRunner().invoke(
            pipelines.main,
            [
                *args,
                "--target",
                "cuda:sm_100a",
                "--prefill-token-budget",
                "64",
                "--prefill-graph-max-tokens",
                "32",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        assert events == [
            ("api", "cuda"),
            ("arch", "sm_100a"),
            ("count", 2),
            ("export", (64, 32)),
        ]
    finally:
        set_count(0)


def test_export_refuses_an_existing_output(tmp_path: Path) -> None:
    (tmp_path / "taken.m").mkdir()
    result = CliRunner().invoke(
        pipelines.main, _export_args(tmp_path / "taken"), catch_exceptions=False
    )
    assert result.exit_code == 1
    assert "already exists" in result.output


def test_export_help_lists_pipeline_options() -> None:
    result = CliRunner().invoke(pipelines.main, ["export", "--help"])
    assert result.exit_code == 0, result.output
    assert "--output" in result.output
    assert "--model-path" in result.output
    assert "--target" in result.output
    assert "--prefill-token-budget" in result.output
    assert "--prefill-graph-max-tokens" in result.output
