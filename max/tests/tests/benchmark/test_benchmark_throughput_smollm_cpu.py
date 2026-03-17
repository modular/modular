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
import os
import subprocess
import sys

import hf_repo_lock
from max.pipelines.lib import generate_local_model_path

REPO_ID = "HuggingFaceTB/SmolLM-135M"
REPO_REVISION = hf_repo_lock.revision_for_hf_repo(REPO_ID)

logger = logging.getLogger("max.pipelines")


def test_python_serving_cpu() -> None:
    assert isinstance(REPO_REVISION, str), (
        "REPO_REVISION must be a string and present in hf-repo-lock.tsv"
    )
    try:
        model_path = generate_local_model_path(REPO_ID, REPO_REVISION)
    except FileNotFoundError as e:
        logger.warning(f"Failed to generate local model path: {str(e)}")
        logger.warning(
            f"Falling back to repo_id: {REPO_ID} as config to PipelineConfig"
        )
        model_path = REPO_ID

    smoke_cmd = [
        os.environ["BENCHMARK_THROUGHPUT_BINARY"],
        f"--pipeline.model.model-path={model_path}",
        "--pipeline.model.quantization-encoding=float32",
        "--other.num-prompts=2",
        "--pipeline.runtime.max-batch-size=1",
        "--input-len=108",
        "--output-len=50",
        "--other.trust-remote-code",
        "--pipeline.model.max-length=512",
    ]
    process_result = subprocess.run(
        smoke_cmd,
        stdout=subprocess.PIPE,
        env=dict(os.environ),
    )
    if process_result.returncode != 0:
        print(
            "Error: Subprocess run exited with status code",
            process_result.returncode,
            file=sys.stderr,
        )
        print(
            "Command that failed:" + str(smoke_cmd),
            file=sys.stderr,
        )
        print("Command's stdout:", file=sys.stderr)
        print(process_result.stdout.decode("utf-8"), file=sys.stderr)
    assert process_result.returncode == 0
