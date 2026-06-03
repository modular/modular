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

"""Validation helpers for generated token IDs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.pipelines.modeling.types import RequestID


def validate_context_generated_tokens(
    token_ids: Sequence[int],
    *,
    vocab_size: int | None,
    request_id: RequestID | None = None,
) -> None:
    """Raise if any generated token is negative or outside the vocabulary."""
    for token_id in token_ids:
        if token_id < 0:
            request_suffix = (
                f" request_id={request_id}" if request_id is not None else ""
            )
            raise RuntimeError(
                f"Generated negative token_id={token_id}{request_suffix}"
            )
        if vocab_size is not None and token_id >= vocab_size:
            request_suffix = (
                f" request_id={request_id}" if request_id is not None else ""
            )
            raise RuntimeError(
                "Generated out-of-vocabulary token_id="
                f"{token_id} (valid range: [0, {vocab_size}))"
                f"{request_suffix}"
            )
