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
"""JSON backend."""

from __future__ import annotations

import json
from collections.abc import Iterable

from ..backend import Backend, indent
from ..types import Entry


class JsonBackend(Backend):
    def generate_all(self, entries: Iterable[Entry]):  # noqa: ANN201
        entries = list(entries)
        yield "["
        for i, entry in enumerate(entries):
            raw = json.dumps(entry.data)
            if i < len(entries) - 1:
                raw += ","
            yield indent(raw, 1)
        yield "]"
