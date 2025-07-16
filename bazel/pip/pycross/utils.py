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

from typing import Any


def assert_keys(
    blob: dict[str, Any], *, required: set[str], optional: set[str]
) -> None:
    """Validate only known keys exist in the blob and that required keys are present to produce better errors."""
    extra_keys = set(blob.keys()) - optional - required
    if extra_keys:
        raise ValueError(f"Unexpected keys in blob: {extra_keys}, {blob}")

    missing_keys = required - blob.keys()
    if missing_keys:
        raise ValueError(
            f"Missing required keys in blob: {missing_keys}, {blob}"
        )
