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

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from max.driver import DeviceSpec


@dataclass
class CLITestCommand:
    args: list[str]
    expected: dict[str, Any]
    valid: bool


class CLITestEnum(str, Enum):
    DEFAULT = "default"
    ALT = "alt"


@dataclass
class CLITestConfig:
    bool_field: bool = False
    enum_field: CLITestEnum = CLITestEnum.DEFAULT
    path_sequence_field: list[Path] = field(default_factory=list)
    device_specs_field: list[DeviceSpec] = field(
        default_factory=lambda: [DeviceSpec.cpu()]
    )
    optional_str_field: str | None = None
    optional_enum_field: CLITestEnum | None = None


@dataclass
class Output:
    default: Any
    field_type: Any
    flag: bool
    multiple: bool
    optional: bool
