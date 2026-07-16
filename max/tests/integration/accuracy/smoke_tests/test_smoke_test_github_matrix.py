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

from click.testing import CliRunner
from hf_repo_lock import load_db
from smoke_tests import smoke_test, smoke_test_github_matrix


def test_all_models_in_hf_repo_lock() -> None:
    """Every smoke test model must have a pinned revision in hf-repo-lock.tsv."""
    lock = load_db()
    missing = [m for m in smoke_test_github_matrix.HF_MODELS if m not in lock]
    assert not missing, f"Models missing from hf-repo-lock.tsv: {missing}"


def test_custom_model_keys_have_dunder() -> None:
    bad = [k for k in smoke_test_github_matrix.CUSTOM_MODELS if "__" not in k]
    assert not bad, (
        f"CUSTOM_MODELS keys must contain '__' to separate the base model "
        f"from the alias suffix: {bad}"
    )


def test_custom_models_defined_in_model_aliases() -> None:
    missing = [
        k
        for k in smoke_test_github_matrix.CUSTOM_MODELS
        if k not in smoke_test.MODEL_RECIPES
    ]
    assert not missing, (
        f"CUSTOM_MODELS keys must have a corresponding entry in "
        f"smoke_test.MODEL_RECIPES: {missing}"
    )


def test_model_aliases_in_custom_models() -> None:
    missing = [
        k
        for k in smoke_test.MODEL_RECIPES
        if "__" in k and k not in smoke_test_github_matrix.CUSTOM_MODELS
    ]
    assert not missing, (
        f"Custom MODEL_RECIPES keys must have a corresponding entry in "
        f"smoke_test_github_matrix.CUSTOM_MODELS: {missing}"
    )


def test_smoke_test_github_matrix_b200_max_ci() -> None:
    runner = CliRunner()
    result = runner.invoke(
        smoke_test_github_matrix.main,
        ["--framework", "max-ci", "--run-on-b200"],
    )

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert "include" in output
    assert len(output["include"]) > 0
