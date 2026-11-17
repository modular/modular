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
from max.tests.integration.accuracy.logit_verification.logit_verification_config import (
    LOGIT_VERIFICATION_CONFIG,
)
from max.tests.integration.tools.create_pipelines import PIPELINE_ORACLES


def test_every_configured_pipeline_has_an_oracle() -> None:
    """A `pipeline` with no oracle entry runs without its `config_params`.

    `generate_llm_logits` falls back to a bare `GenericOracle` on a lookup
    miss, so a drifted `pipeline` string still loads and still produces
    numbers -- just not from the run the entry describes.
    """
    missing = sorted(
        {
            entry.pipeline
            for entry in LOGIT_VERIFICATION_CONFIG.pipelines.values()
            if entry.pipeline not in PIPELINE_ORACLES
        }
    )
    assert not missing, (
        f"logit verification entries whose `pipeline` is not a"
        f" PIPELINE_ORACLES key: {missing}"
    )
