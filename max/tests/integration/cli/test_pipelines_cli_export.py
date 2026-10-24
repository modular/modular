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

import json
import os
from pathlib import Path

from click.testing import CliRunner
from max._artifact.batch_processor import BatchProcessorDims
from max._artifact.serialization import deserialize
from max._entrypoints import pipelines


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

    section = json.loads((dest / "artifact.json").read_text())["model"]
    assert set(section) == {"decoder", "sampler"}
    assert section["decoder"]["mef"] == {"$file": "decoder/model.mef"}
    loaded = deserialize(dest).model
    assert loaded is not None
    assert set(loaded.roles) == {"decoder", "sampler"}


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
