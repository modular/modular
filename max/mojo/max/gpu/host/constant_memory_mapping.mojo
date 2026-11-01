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
"""This module provides functionality for mapping constant memory between host and device.

The module includes the `ConstantMemoryMapping` struct which represents a mapping of
constant memory that can be used for efficient data transfer between host and GPU device.
"""


@__doc_inline
from ._legacy.constant_memory_mapping import (
    ConstantMemoryMapping,
)
