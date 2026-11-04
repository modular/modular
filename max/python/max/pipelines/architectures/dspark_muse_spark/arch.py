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
"""The ``MuseSparkDSparkModel`` draft architecture."""

from dataclasses import replace

from max.pipelines.lib import SupportedArchitecture

from ..dspark_draft.arch import dspark_speculators_draft_arch
from .model_config import MuseSparkDSparkDraftArchConfig

# The drafter is only served as a --draft-model beside a Muse Spark target.
muse_spark_dspark_draft_arch: SupportedArchitecture = replace(
    dspark_speculators_draft_arch,
    name="MuseSparkDSparkModel",
    example_repo_ids=["Inferact/Muse-Spark-Open-DSpark"],
    config=MuseSparkDSparkDraftArchConfig,
    # The pairing places it on the target's mesh.
    multi_gpu_supported=True,
)
