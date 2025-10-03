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

from __future__ import annotations

import os

from .interface import BenchmarkDataset
from .types import DatasetMode


class LocalBenchmarkDataset(BenchmarkDataset):
    """Abstract base class for local benchmark datasets.

    This class provides a common interface for working with local benchmark datasets.
    It handles automatic dataset fetching and provides a standardized interface for sampling requests.
    """

    dataset_mode = DatasetMode.LOCAL

    def fetch(self) -> None:
        # For local mode, dataset_path should already be set and validated
        if self.dataset_path is None:
            raise ValueError("For LOCAL mode, dataset_path must be provided")
        if not os.path.exists(self.dataset_path):
            raise ValueError(
                f"Local dataset path {self.dataset_path} does not exist"
            )
