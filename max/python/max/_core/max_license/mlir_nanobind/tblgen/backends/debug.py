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
"""Debug backend."""

from __future__ import annotations

from collections.abc import Iterable

from ..backend import Backend
from ..types import Entry


class DebugBackend(Backend):
    def generate_all(self, entries: Iterable[Entry]):  # noqa: ANN201
        for entry in entries:
            yield f"/// {entry.name}"
            yield repr(entry)
            yield ""
