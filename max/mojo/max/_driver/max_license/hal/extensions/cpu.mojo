# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
from std.ffi import _DLHandle

from ..status import HALError
from .traits import HALExtensions


struct CPUHALExtensions(HALExtensions):
    """`HALExtensions` implementation for the host CPU."""

    def __init__(out self, lib: _DLHandle, so_path: StringSpan) raises HALError:
        pass
