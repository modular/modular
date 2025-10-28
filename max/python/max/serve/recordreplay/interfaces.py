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

"""Interfaces for interacting with record/replay structures."""

from __future__ import annotations

import abc

from .schema import RecordingItem

__all__ = [
    # Please keep this list alphabetized.
    "Recorder",
]


class Recorder(abc.ABC):
    """Something that can accept recording items to be persisted somewhere."""

    @abc.abstractmethod
    def record(self, item: RecordingItem) -> None:
        """Record an item."""
        ...
