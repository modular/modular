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

import importlib
import sys
from types import ModuleType

import pytest
from max.pipelines.lib.registry import PipelineRegistry


def test_import_custom_architectures_requires_architectures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = ModuleType("empty_arch")
    monkeypatch.setattr(importlib, "import_module", lambda _: module)
    monkeypatch.setattr(sys, "path", sys.path.copy())

    with pytest.raises(
        ValueError,
        match=r"did not expose an `ARCHITECTURES` list\. Module: empty_arch",
    ):
        PipelineRegistry([])._import_custom_architectures(["empty_arch"])
