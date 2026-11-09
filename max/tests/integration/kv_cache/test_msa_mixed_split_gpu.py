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
"""Mixed-step split of the MSA ops: the presubmit BF16 and FP8 harnesses."""

from msa_mixed_split_cases import MixedSplitCases, harness_fixture


class TestMsaMixedSplit(MixedSplitCases):
    harness = harness_fixture("bf16_1h", "fp8_4h")
