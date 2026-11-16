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

import pytest
from click.testing import CliRunner
from max._entrypoints import pipelines


def test_pipelines_list(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        pipelines.main(["list"])
    captured = capsys.readouterr()
    assert "DeepseekV2ForCausalLM" in captured.out


@pytest.mark.parametrize(
    ("command", "refusal"),
    [
        (["serve"], "not max serve"),
        (["warm-cache"], "no decoder to warm"),
        (["export", "--output", "unused"], "nothing to export"),
    ],
)
def test_commands_that_compile_refuse_eager(
    command: list[str], refusal: str
) -> None:
    result = CliRunner().invoke(
        pipelines.main,
        [*command, "--model-path", "HuggingFaceTB/SmolLM2-135M", "--eager"],
    )
    assert result.exit_code == 2, result.output
    assert refusal in result.output
