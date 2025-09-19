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

"""GPU diagnostics API.

This module allows accessing information about GPU(s) on the system.
Information can be accessed synchronously or collected in the background for
later retrieval.
"""

from ._bgrec import BackgroundRecorder as BackgroundRecorder
from ._multi import GPUDiagContext as GPUDiagContext
from ._types import GPUStats as GPUStats
from ._types import MemoryStats as MemoryStats
from ._types import UtilizationStats as UtilizationStats
