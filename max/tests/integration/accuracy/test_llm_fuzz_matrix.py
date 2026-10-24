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

import json

import llm_fuzz_matrix
import pytest
from click.testing import CliRunner


def test_llm_fuzz_matrix_schedule_all() -> None:
    """Schedule event returns all pipelines."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        ["--event-name", "schedule"],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "include" in output
    assert len(output["include"]) == len(llm_fuzz_matrix.PIPELINES)


@pytest.mark.parametrize(
    "pipeline",
    [
        "minimax/MiniMax-M3-MXFP8-ep-tp",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4__dspark-tp2",
    ],
)
def test_llm_fuzz_matrix_dispatch_specific(pipeline: str) -> None:
    """workflow_dispatch with a specific pipeline returns only that one."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        [
            "--event-name",
            "workflow_dispatch",
            "--selected-pipeline",
            pipeline,
        ],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert len(output["include"]) == 1
    assert output["include"][0]["pipeline"] == pipeline


def test_llm_fuzz_matrix_dispatch_all() -> None:
    """workflow_dispatch with 'all' returns every entry."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        ["--event-name", "workflow_dispatch", "--selected-pipeline", "all"],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert len(output["include"]) == len(llm_fuzz_matrix.PIPELINES)


def test_llm_fuzz_matrix_dispatch_unknown_pipeline() -> None:
    """workflow_dispatch with an unknown pipeline exits non-zero."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        [
            "--event-name",
            "workflow_dispatch",
            "--selected-pipeline",
            "does-not-exist",
        ],
    )
    assert result.exit_code != 0


def test_llm_fuzz_matrix_local_model_path_override() -> None:
    """--local-model-path reroutes every entry to the supplied runner."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        [
            "--event-name",
            "workflow_dispatch",
            "--selected-pipeline",
            "all",
            "--local-model-path",
            "/mnt/local/data/test/",
            "--local-model-runner",
            "test-mount-runner",
            "--local-model-instance-type",
            "test-instance",
        ],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["include"]
    for entry in output["include"]:
        assert entry["runner"] == "test-mount-runner"
        assert entry["instance_type"] == "test-instance"
        assert entry["model_path"] == "/mnt/local/data/test/"


def test_llm_fuzz_matrix_local_model_path_requires_runner() -> None:
    """--local-model-path without --local-model-runner exits non-zero."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        [
            "--event-name",
            "workflow_dispatch",
            "--selected-pipeline",
            "all",
            "--local-model-path",
            "/mnt/local/data/test/",
        ],
    )
    assert result.exit_code != 0


def test_llm_fuzz_matrix_no_local_path_keeps_default_runner() -> None:
    """Without --local-model-path each entry keeps its default runner."""
    runner = CliRunner()
    result = runner.invoke(
        llm_fuzz_matrix.main,
        ["--event-name", "workflow_dispatch", "--selected-pipeline", "all"],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    runners = {entry["runner"] for entry in output["include"]}
    # Default runners are open-weights only; no override should leak in.
    for default_runner in runners:
        assert "modrunner-" in default_runner


def test_smoke_test_pipeline_is_in_the_matrix() -> None:
    """The PR fallback must name an entry that exists.

    filter_entries selects the fallback with a comprehension over PIPELINES,
    so a name no entry carries yields an empty matrix rather than an error:
    configure succeeds, evaluate is skipped by its matrix check, and the run
    is green having fuzzed nothing. Removing the pipeline this constant named
    is exactly how that happens.
    """
    assert llm_fuzz_matrix.SMOKE_TEST_PIPELINE in {
        entry.pipeline for entry in llm_fuzz_matrix.PIPELINES
    }
