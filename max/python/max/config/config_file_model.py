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
"""ConfigFileModel for Pydantic-based config classes."""

from __future__ import annotations

import enum
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel, Field, model_validator


class ConfigFileModel(BaseModel):
    """Base class for models that can load configuration from a file.

    This class provides functionality for Pydantic-based config classes to load
    configuration from YAML files. Config classes should inherit from this class
    to enable config file support.

    Example:
        ```python
        from max.config import ConfigFileModel
        from pydantic import Field

        class MyConfig(ConfigFileModel):
            value: int = Field(default=1)

        # Can be used with --config-file config.yaml
        config = MyConfig(config_file="config.yaml")
        ```
    """

    config_file: str | None = None
    """Path to the configuration file."""

    # TODO: This is very similar to _config_file_section_name in MAXConfig.
    # We'll deprecate that and use this instead in the future once we've fully
    # migrated to cyclopts for our CLI bindings.
    section_name: str | None = Field(default=None, exclude=True)
    """Optional section name for comprehensive/multi-section config files.

    If not provided, values are loaded from the YAML top-level (treating the file
    as an "individual config" file).
    """

    @model_validator(mode="before")
    @classmethod
    def load_config_file(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Load configuration from YAML file if config_file is provided.

        This validator runs before Pydantic validation. Cyclopts processes config
        sources in order: CLI args are parsed first, then env vars (from Env config
        source) are applied. When this validator runs, `data` already contains CLI
        args and env vars merged together.

        To achieve the correct precedence (CLI > Config File > Env Vars > Defaults),
        we need to separate CLI args from env vars. However, since cyclopts merges
        them before validation, we approximate by:
        1. Loading config file values
        2. Merging with data (CLI args + env vars), where data takes precedence

        This results in: CLI args > Env vars > Config file > Defaults

        Note: The README documents the desired precedence, but due to cyclopts'
        architecture, config files cannot override env vars while still allowing
        CLI args to override everything. The actual precedence is:
        1. CLI arguments (highest)
        2. Environment variables
        3. Config files
        4. Defaults (lowest)

        Args:
            data: Dictionary of data to validate, may contain 'config_file' key.
                  This dict already contains CLI args and env vars merged by cyclopts.
        Returns:
            Dictionary with config file values merged in if config_file was provided.
        """
        if (config_file := data.get("config_file")) is not None:
            with open(config_file) as f:
                loaded_data = yaml.safe_load(f) or {}

            if not isinstance(loaded_data, dict):
                raise TypeError(
                    f"Configuration file must contain a mapping at the root, got {type(loaded_data)}"
                )

            explicit_section_name = data.get("section_name")
            if explicit_section_name is not None:
                # Caller explicitly requested a section.
                section_data = loaded_data.get(explicit_section_name)
                if section_data is None:
                    available_sections = [
                        key
                        for key, value in loaded_data.items()
                        if isinstance(value, dict)
                    ]
                    raise KeyError(
                        f"Section '{explicit_section_name}' not found in configuration file. "
                        f"Available sections: {available_sections}."
                    )
                if not isinstance(section_data, dict):
                    raise TypeError(
                        f"Section '{explicit_section_name}' must be a mapping, got {type(section_data)}"
                    )
                loaded_data = section_data

            # TODO: This call is only needed for DType since it's not a (str, enum) type.
            loaded_data = cls._coerce_enum_fields(loaded_data)

            # Merge: config file values are loaded, then overridden by CLI args + env vars.
            # Note: Due to cyclopts processing order, env vars override config files.
            data = loaded_data | data
        return data

    # TODO(SERVSYS-1087): If still needed, move them to a separate class.
    # Doesn't seem specific to ConfigFileModel.
    def __eq__(self, other: object) -> bool:
        """Structural equality based on public model fields.

        Pydantic's default equality can incorporate internal bookkeeping (e.g.
        fields-set tracking and private attributes). For config objects we want a
        stable notion of equality based on their declared fields, so config
        roundtrips through pickling/serialization compare equal.
        """
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.model_dump(mode="python") == other.model_dump(mode="python")

    def __ne__(self, other: object) -> bool:
        """Negation of `__eq__` with consistent NotImplemented behavior."""
        eq_result = self.__eq__(other)
        if eq_result is NotImplemented:
            return NotImplemented
        return not eq_result

    @classmethod
    def _coerce_enum_fields(cls, loaded_data: dict[str, Any]) -> dict[str, Any]:
        """Coerce string values in loaded YAML into Enum members when possible.

        Pydantic's default enum parsing prefers enum *values* (often integers),
        which is inconvenient for config files where users commonly provide the
        enum *name* (e.g. "float16"). For enum-typed fields, if the YAML provides
        a string, try to map it to a member name case-insensitively.
        """
        for field_name, field_info in cls.model_fields.items():
            if field_name not in loaded_data:
                continue
            value = loaded_data[field_name]
            if not isinstance(value, str):
                continue

            enum_type = cls._extract_enum_type(field_info.annotation)
            if enum_type is None:
                continue

            value_casefolded = value.casefold()
            for enum_member in enum_type:
                if enum_member.name.casefold() == value_casefolded:
                    loaded_data[field_name] = enum_member
                    break
            else:
                # Fall back to the Enum's constructor, which may have custom
                # string handling (e.g. DType supports MLIR strings like "f16").
                try:
                    loaded_data[field_name] = enum_type(value)
                except Exception:
                    pass

        return loaded_data

    @staticmethod
    def _extract_enum_type(annotation: Any) -> type[enum.Enum] | None:
        """Extract an Enum type from a field annotation, handling Optional/Union."""
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return annotation

        origin = get_origin(annotation)
        if origin is None:
            return None

        args = get_args(annotation)
        for arg in args:
            if isinstance(arg, type) and issubclass(arg, enum.Enum):
                return arg

        return None
