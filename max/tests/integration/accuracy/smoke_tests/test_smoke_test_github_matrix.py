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
from smoke_tests import smoke_test_github_matrix


def test_model_keys_are_lowercase() -> None:
    bad = [k for k in smoke_test_github_matrix.MODELS if k != k.lower()]
    assert not bad, f"Model keys must be lowercase: {bad}"


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
