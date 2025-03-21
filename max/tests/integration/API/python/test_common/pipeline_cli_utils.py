# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

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
    optional_str_field: Optional[str] = None
    optional_enum_field: Optional[CLITestEnum] = None


@dataclass
class Output:
    default: Any
    field_type: Any
    flag: bool
    multiple: bool
    optional: bool
