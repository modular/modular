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
"""Compatibility layer that re-exports ArcPointer utilities.

Existing imports such as `from memory.arc import ArcPointer` continue to work
by re-exporting everything from `arc_pointer`.
"""

# Re-export every symbol defined in arc_pointer for backwards compatibility.
from .arc_pointer import *
