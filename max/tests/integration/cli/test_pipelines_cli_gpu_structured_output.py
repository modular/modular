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

import logging

import hf_repo_lock
import pytest
from max.entrypoints import pipelines
from test_common.graph_utils import is_h100_h200

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


@pytest.mark.skipif(is_h100_h200(), reason="AITLIB-342: Failing on H100")
def test_pipelines_cli__smollm_bfloat16_with_structured_output_enabled(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert isinstance(REVISION, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    local_model_path = REPO_ID

    with pytest.raises(SystemExit):
        pipelines.main(
            [
                "generate",
                "--model-path",
                local_model_path,
                "--prompt",
                "Why is the sky blue",
                "--trust-remote-code",
                "--device-memory-utilization=0.1",
                "--quantization-encoding=bfloat16",
                "--devices=gpu",
                "--huggingface-model-revision",
                REVISION,
                "--huggingface-weight-revision",
                REVISION,
                # Enabling structured output server-wide without a JSON schema
                # must not change the outputs of the base chat experience.
                "--enable-structured-output",
                "--top-k=1",
            ]
        )
    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert any(
        word in captured.out.lower()
        for word in ["light", "scatter", "atmosphere", "blue"]
    )
