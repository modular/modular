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

from typing import Any

import yaml
from pydantic import BaseModel, model_validator


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
        if "config_file" in data:
            with open(data["config_file"]) as f:
                loaded_data = yaml.safe_load(f) or {}
                # Merge: config file values are loaded, then overridden by CLI args + env vars
                # Note: Due to cyclopts processing order, env vars override config files
                data = loaded_data | data
        return data
