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

"""The pipeline flags shared by the structured-output CLI test and its precompiler.

The precompiled MEFs are matched to the graphs the GPU run builds, so both sides
import ``pipeline_flags`` and a flag cannot be changed on one side only.
"""

from __future__ import annotations

import hf_repo_lock

REPO_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


def pipeline_flags() -> list[str]:
    """Returns the pipeline-config flags for the structured-output run.

    These are the flags that shape the compiled graphs, so the same list drives
    ``generate`` on the GPU and ``warm-cache`` on the CPU precompiler.
    Per-request sampling params (``--top-k`` and friends) are deliberately
    absent: they reach the pipeline as a ``SamplingParams`` per request rather
    than as graph structure, and ``warm-cache`` does not accept them.

    Returns:
        The CLI flags, with the locked HuggingFace revision filled in.
    """
    revision = hf_repo_lock.revision_for_hf_repo(REPO_ID)
    assert isinstance(revision, str), (
        "REVISION must be a string and present in hf-repo-lock.tsv"
    )
    return [
        "--model-path",
        REPO_ID,
        "--trust-remote-code",
        "--device-memory-utilization=0.1",
        "--quantization-encoding=bfloat16",
        "--devices=gpu",
        "--huggingface-model-revision",
        revision,
        "--huggingface-weight-revision",
        revision,
        # Enabling structured output server-wide without a JSON schema must not
        # change the outputs of the base chat experience.
        "--enable-structured-output",
    ]
