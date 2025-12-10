# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Unit test to test generate_llm_logits functionality"""

from pathlib import Path

import generate_llm_logits
from click.testing import CliRunner


def test_generate_llm_logits_smollm(tmp_path: Path) -> None:
    runner = CliRunner()
    output_file = tmp_path / "output_goldens.json"
    result = runner.invoke(
        generate_llm_logits.main,
        [
            "--framework=max",
            "--pipeline=HuggingFaceTB/SmolLM2-135M",
            "--encoding=bfloat16",
            "--device=gpu",
            "--output",
            f"{output_file}",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.stdout_bytes.decode("utf-8")
    assert output_file.exists()
