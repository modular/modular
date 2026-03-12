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
"""Dropout module (identity at inference time)."""

from __future__ import annotations

from max.experimental.tensor import Tensor

from .module import Module


class Dropout(Module[[Tensor], Tensor]):
    """Dropout layer — a no-op at inference / compilation time.

    During inference (the only mode supported by MAX graph compilation),
    dropout is never applied; the input is returned unchanged.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x
