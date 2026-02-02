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
"""Function for verifying that the predictions match the references."""

from typing import Any


def results_match(
    references: list[str], predictions: list[str], **kwargs: Any
) -> int:
    total = 0
    for reference, prediction in zip(references, predictions, strict=True):
        total += reference in prediction
    return total
