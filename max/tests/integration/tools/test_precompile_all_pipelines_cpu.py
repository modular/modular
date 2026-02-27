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
"""Test precompile-all-pipelines orchestrator compiles mpnet on CPU."""

from precompile_all_pipelines import (
    collect_precompile_jobs,
    run_precompile_inprocess,
)
from verify_pipelines import TagFilter


def test_precompile_mpnet() -> None:
    jobs = collect_precompile_jobs(
        devices="cpu",
        target=None,
        tag_filter=TagFilter(),
        name_filter="mpnet",
    )
    assert len(jobs) >= 1, (
        "No jobs selected -- mpnet may have been removed from "
        "LOGIT_VERIFICATION_CONFIG or lost DeviceKind.CPU compatibility"
    )

    success, output, _ = run_precompile_inprocess(jobs[0])
    assert success, f"precompile compilation failed:\n{output}"
