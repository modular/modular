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

import dataset_eval_matrix
from click.testing import CliRunner


def test_dataset_eval_matrix_schedule_all() -> None:
    """Smoke test: schedule event returns all pipelines."""
    runner = CliRunner()
    result = runner.invoke(
        dataset_eval_matrix.main,
        ["--event-name", "schedule"],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "include" in output
    assert len(output["include"]) == len(dataset_eval_matrix.PIPELINES)


def test_dataset_eval_matrix_dispatch_specific() -> None:
    """workflow_dispatch with a specific pipeline returns only that one."""
    runner = CliRunner()
    result = runner.invoke(
        dataset_eval_matrix.main,
        [
            "--event-name",
            "workflow_dispatch",
            "--selected-pipeline",
            "sentence-transformers/all-mpnet-base-v2",
        ],
    )
    assert result.exit_code == 0
    output = json.loads(result.output)
    assert len(output["include"]) == 1
    assert (
        output["include"][0]["pipeline"]
        == "sentence-transformers/all-mpnet-base-v2"
    )
