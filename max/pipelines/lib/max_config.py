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
"""MAX config classes."""

from __future__ import annotations

import argparse
import enum
import logging
import os
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import yaml
from max.dtype import DType
from max.engine import GPUProfilingMode
from max.nn.kv_cache import KVCacheStrategy
from max.serve.queue.zmq_queue import generate_zmq_ipc_path

logger = logging.getLogger("max.pipelines")

T = TypeVar("T", bound="MAXConfig")

STRINGLY_TYPED_ENUM_MAPPING: Mapping[str, type[enum.Enum]] = {
    "DType": DType,
    "GPUProfilingMode": GPUProfilingMode,
    "KVCacheStrategy": KVCacheStrategy,
}

STRINGLY_TYPED_BASIC_MAPPING: Mapping[str, type] = {
    "bool": bool,
    "float": float,
    "int": int,
    "str": str,
}

MAX_CONFIG_METADATA_FIELDS: Mapping[str, type[Any]] = {
    "name": str,
    "description": str,
    "version": str,
    "depends_on": str,
}


# TODO: I believe we have some utils like this in our entrypoint code. We should
# move and consolidate them.
def _get_argparse_type_and_action(
    field_type: Any, field_default: Any
) -> tuple[Any, str | type[argparse.Action] | None]:
    """Determine the appropriate argparse type and action for a field type.

    This reuses the same type analysis logic as convert_max_config_value but
    returns argparse-compatible type and action parameters.

    Args:
        field_type: The field type to analyze
        field_default: The default value for the field

    Returns:
        Tuple of (type_func, action) where:
        - type_func: The type function to pass to add_argument() or None for action args
        - action: The action string for add_argument() or None for typed args
    """

    # Get the origin and args for generic types (e.g., Optional[int] -> Union, (int, NoneType))
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types (which are Union[T1, T2, ...])
    if origin is Union:
        # Check if this is Optional[T] (Union[T, None])
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], get the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return _get_argparse_type_and_action(non_none_type, field_default)
        else:
            # Complex Union - default to string
            return str, None
    elif origin is list:
        if args:
            # For List[T], the parser should convert each element to T
            element_type = args[0]
            if element_type in (int, float, str):
                return element_type, None
            elif isinstance(element_type, type) and issubclass(
                element_type, enum.Enum
            ):
                return str, None  # Enums converted from strings
            else:
                return str, None
        else:
            return str, None

    # Handle basic types
    if field_type in (int, float, str):
        return field_type, None

    # Handle enum types
    elif isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return str, None  # Enums are parsed as strings then converted

    # Handle boolean conversion
    elif field_type is bool:
        return None, argparse.BooleanOptionalAction

    # Handle modern list syntax (list[T]) - fallback for types not caught above
    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        if hasattr(field_type, "__args__") and field_type.__args__:
            element_type = field_type.__args__[0]
            if element_type in (int, float, str):
                return element_type, None
            elif isinstance(element_type, type) and issubclass(
                element_type, enum.Enum
            ):
                return str, None  # Enums converted from strings
            else:
                return str, None
        else:
            return str, None

    # Default to string for unknown types
    return str, None


def convert_max_config_value(
    value: Any, field_type: Any, field_name: str
) -> Any:
    """Convert a config value to the appropriate type.

    Handles enums, Optional types, Union types, lists, and basic types.

    Args:
        value: The value from the configuration file.
        field_type: The expected type of the field.
        field_name: The name of the field (for error messages).

    Returns:
        The converted value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    # Handle None values
    if value is None:
        return None

    # Get the origin and args for generic types (e.g., Optional[int] -> Union, (int, NoneType))
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Handle Optional types (which are Union[T1, T2, ...])
    if origin is Union:
        # Check if this is Optional[T] (Union[T, None])
        if len(args) == 2 and type(None) in args:
            # This is Optional[T], get the non-None type
            non_none_type = args[0] if args[1] is type(None) else args[1]
            if value is None:
                return None
            # Recursively convert using the non-None type
            return convert_max_config_value(
                value=value,
                field_type=non_none_type,
                field_name=field_name,
            )
        else:
            # This is a more complex Union, try each type until one works
            last_error = None
            for arg_type in args:
                try:
                    return convert_max_config_value(
                        value=value,
                        field_type=arg_type,
                        field_name=field_name,
                    )
                except (ValueError, TypeError) as e:
                    last_error = e
                    continue
            # If none of the union types worked, raise the last error
            if last_error:
                raise ValueError(
                    f"Cannot convert '{value}' to any type in {field_type}: {last_error}"
                )
            else:
                raise ValueError(f"Cannot convert '{value}' to {field_type}")
    # Handle List types
    elif origin is list:
        if not isinstance(value, list):
            raise ValueError(
                f"Expected list for field '{field_name}', got {type(value)}"
            )

        # If we have type args, convert each element
        if args:
            element_type = args[0]
            return [
                convert_max_config_value(
                    value=item,
                    field_type=element_type,
                    field_name=f"{field_name}[{i}]",
                )
                for i, item in enumerate(value)
            ]
        else:
            # No type information, return as-is
            return value

    # Handle string-typed fields.
    if isinstance(field_type, str):
        if field_type in STRINGLY_TYPED_ENUM_MAPPING:
            field_type = STRINGLY_TYPED_ENUM_MAPPING[field_type]
        elif field_type in STRINGLY_TYPED_BASIC_MAPPING:
            field_type = STRINGLY_TYPED_BASIC_MAPPING[field_type]

    # Handle basic types
    if field_type in (int, float, str):
        try:
            return field_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert '{value}' to {field_type.__name__} for field '{field_name}': {e}"
            ) from e
    # Handle enum types
    elif isinstance(field_type, type) and issubclass(field_type, enum.Enum):
        return _convert_enum_value(
            value=value, enum_type=field_type, field_name=field_name
        )
    # Handle boolean conversion
    elif field_type is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ("true", "1", "yes", "on"):
                return True
            elif lower_value in ("false", "0", "no", "off"):
                return False
            else:
                raise ValueError(
                    f"Cannot convert '{value}' to bool for field '{field_name}': "
                    f"expected 'true'/'false', '1'/'0', 'yes'/'no', or 'on'/'off'"
                )
        try:
            # Handle numeric values (0/1)
            if isinstance(value, (int, float)):
                return bool(value)
            else:
                # Try to convert to int first, then to bool
                return bool(int(value))
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Cannot convert '{value}' to bool for field '{field_name}': {e}"
            ) from e

    # Handle modern list syntax (list[T]) - fallback for types not caught above
    if hasattr(field_type, "__origin__") and field_type.__origin__ is list:
        # This handles list[str], list[int], etc. in Python 3.9+
        if hasattr(field_type, "__args__") and field_type.__args__:
            element_type = field_type.__args__[0]
            if not isinstance(value, list):
                raise ValueError(
                    f"Expected list for field '{field_name}', got {type(value)}"
                )
            return [
                convert_max_config_value(
                    value=item,
                    field_type=element_type,
                    field_name=f"{field_name}[{i}]",
                )
                for i, item in enumerate(value)
            ]
        else:
            # No type information, return as-is
            return value if isinstance(value, list) else [value]

    raise ValueError(f"Unexpected field type: {field_type}")


def _convert_enum_value(
    value: Any, enum_type: type[enum.Enum], field_name: str
) -> enum.Enum:
    """Convert a value to an enum type (case-insensitive).

    Args:
        value: The value to convert.
        enum_type: The enum type to convert to.
        field_name: The field name for error messages.

    Returns:
        The converted enum value.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, enum_type):
        return value
    elif isinstance(value, str):
        value_casefolded = value.casefold()
        # Try to get the enum by name first (case-insensitive)
        for enum_member in enum_type:
            if enum_member.name.casefold() == value_casefolded:
                return enum_member

    try:
        return enum_type(int(value))
    except ValueError:
        pass

    # If we get here, the conversion failed
    valid_names = [e.name for e in enum_type]
    valid_values = [e.value for e in enum_type]
    raise ValueError(
        f"Invalid enum value '{value}' for {field_name}. "
        f"Valid names: {valid_names}, valid values: {valid_values}"
    )


def get_default_max_config_file_section_name(
    config_class: type[MAXConfig],
) -> str:
    """Get the default section name for a MAXConfig class.

    Args:
        config_class: The config class.

    Returns:
        The default section name.

    Raises:
        ValueError: If the config class doesn't define a _config_file_section_name attribute.
    """
    section_name = getattr(config_class, "_config_file_section_name", None)
    if section_name is None:
        raise ValueError(
            f"Config class {config_class.__name__} must define a '_config_file_section_name' class attribute. "
            f"This attribute specifies the expected configuration section name for this config class."
        )
    return section_name


def deep_merge_max_configs(
    base_config: dict[str, Any], child_config: dict[str, Any]
) -> dict[str, Any]:
    """Deep merge two MAXConfig configuration dictionaries.

    Args:
        base_config: The base MAXConfig configuration dictionary.
        child_config: The child MAXConfig configuration dictionary that takes precedence.

    Returns:
        Merged MAXConfig configuration dictionary with deep merging of nested MAXConfig dictionaries.
    """
    merged = base_config.copy()

    for key, value in child_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            # Both are MAXConfig dictionaries, merge them recursively.
            merged[key] = deep_merge_max_configs(merged[key], value)
        else:
            # Child value takes precedence (either not a dict or key doesn't exist in base).
            merged[key] = value

    return merged


def resolve_max_config_inheritance(
    config_dict: dict[str, Any], config_class: type[MAXConfig]
) -> dict[str, Any]:
    """Resolve configuration inheritance by loading base config and merging.

    Args:
        config_dict: The current configuration dictionary.
        config_class: The config class being loaded.

    Returns:
        Merged configuration dictionary with inheritance resolved.
    """
    depends_on = config_dict.get("depends_on")
    if not depends_on:
        return config_dict

    # Try to load the base config
    try:
        base_config_path = Path(depends_on)
        if not base_config_path.is_absolute():
            raise ValueError(
                f"Relative path inheritance not supported: {depends_on}. "
                f"Please use an absolute path for the 'depends_on' configuration."
            )
        elif not base_config_path.exists():
            raise FileNotFoundError(
                f"Base configuration file not found: {base_config_path}"
            )

        logger.info(f"Loading base configuration from {base_config_path}")

        # Load base config
        with open(base_config_path, encoding="utf-8") as f:
            base_config_dict = yaml.safe_load(f)

        # Keep this assert to help with type checking
        if not isinstance(base_config_dict, dict):
            raise ValueError(
                f"Base configuration file {base_config_path} must contain a dictionary at the top level"
            )

        # Recursively resolve inheritance in base config
        base_config_dict = resolve_max_config_inheritance(
            config_dict=base_config_dict, config_class=config_class
        )

        # Merge base config with current config (current takes precedence)
        return deep_merge_max_configs(base_config_dict, config_dict)

    except (FileNotFoundError, ValueError) as e:
        # Re-raise FileNotFoundError and ValueError exceptions (these are intentional validation errors)
        raise e
    except Exception as e:
        logger.warning(f"Failed to load base configuration '{depends_on}': {e}")
        return config_dict


def _extract_max_config_data(
    config_dict: dict[str, Any],
    config_class: type[MAXConfig],
    section_name: str | None = None,
) -> dict[str, Any]:
    """Extract config data for a specific MAXConfig class from MAXConfig files.

    Args:
        config_dict: The loaded YAML configuration dictionary.
        config_class: The config class we're extracting data for.
        section_name: Optional specific section name to look for.

    Returns:
        Configuration data for the specific config class.

    Raises:
        ValueError: If the appropriate config section cannot be found.
    """
    # Check if this looks like a full MAXConfig file
    # Full MAXConfig files have metadata fields like name, description, version, etc.
    has_metadata = any(
        field in config_dict for field in MAX_CONFIG_METADATA_FIELDS
    )

    # Get class fields to help determine if this is an individual config
    class_fields = {field.name for field in fields(config_class)}

    # Check if the top-level keys are mostly class fields (individual config)
    matching_fields = set(config_dict.keys()) & class_fields

    if not has_metadata and len(matching_fields) > 0:
        # This looks like a partial MAXConfig file
        logger.debug(
            f"Detected partial MAXConfig file for {config_class.__name__}"
        )
        return config_dict

    # This looks like a full MAXConfig file
    logger.debug(f"Detected full MAXConfig file for {config_class.__name__}")

    section_name = section_name or get_default_max_config_file_section_name(
        config_class
    )

    # Handle inheritance via depends_on
    config_dict = resolve_max_config_inheritance(
        config_dict=config_dict, config_class=config_class
    )

    # Look for the appropriate config section
    if section_name in config_dict:
        section_data = config_dict[section_name]
        if isinstance(section_data, dict):
            return section_data
        else:
            raise ValueError(
                f"Section '{section_name}' must be a dictionary, got {type(section_data)}"
            )
    else:
        # Section not found - list available sections for helpful error message
        available_sections = [
            key
            for key in config_dict
            if key not in MAX_CONFIG_METADATA_FIELDS
            and isinstance(config_dict[key], dict)
        ]
        raise ValueError(
            f"Section '{section_name}' not found in configuration file. "
            f"Available sections: {available_sections}. "
            f"Use section_name parameter to specify a different section."
        )


@dataclass
class MAXConfig:
    """Abstract base class for all MAX configs.

    There are some invariants that :obj:`MAXConfig` classes should follow:
    - All config classes should be dataclasses.
    - All config classes should have a :obj:`help()` method that returns a dictionary of config
    options and their descriptions.
    - All config classes dataclass fields should have default values, and hence
    can be trivially initialized via :obj:`cls()`.
    - All config classes should be frozen (except :obj:`KVCacheConfig` for now), to
    avoid accidental modification of config objects.
    - All config classes must have mutually exclusive dataclass fields among
    themselves.
    - All config classes must define a `_config_file_section_name` class attribute specifying
    their expected configuration section name.
    """

    @abstractmethod
    def help(self) -> dict[str, str]:
        """Documentation for this config class. Return a dictionary of config
        options and their descriptions."""
        ...

    @classmethod
    def from_config_file(
        cls: type[T],
        config_path: str | Path,
        section_name: str | None = None,
    ) -> T:
        """Load configuration from a YAML file.

        Supports both individual config files and comprehensive multi-config files.
        For comprehensive files, automatically detects the appropriate section based on class name.

        Args:
            config_path: Path to the YAML configuration file.
            section_name: Optional section name for comprehensive configs.
                         Auto-detected if None.

        Returns:
            Config instance with parameters loaded from the file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the configuration is invalid.

        Examples:
            ```python
            # Individual config file
            config = KVCacheConfig.from_config_file("kv_cache.yaml")

            # Comprehensive config file (auto-detects section)
            kv_config = KVCacheConfig.from_config_file("pipeline.yaml")
            sampling_config = SamplingConfig.from_config_file("pipeline.yaml")

            # Custom section name
            config = KVCacheConfig.from_config_file("config.yaml", "my_cache_section")
            ```
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Load the YAML file.
        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(
                f"Failed to load configuration from {config_path}: {e}"
            ) from e

        # Ensure we have a proper dict[str, Any] type for mypy.
        if not isinstance(config_dict, dict):
            raise ValueError(
                "Configuration file must contain a dictionary at the top level"
            )

        # Extract the config data for this MAXConfig class from the config file.
        config_data = _extract_max_config_data(
            config_dict=config_dict,
            config_class=cls,
            section_name=section_name,
        )

        # Get the dataclass fields for this MAXConfig class.
        class_fields = {field.name: field for field in fields(cls)}

        # Filter config_data to only include valid fields for this MAXConfig class
        filtered_config = {}
        invalid_keys = []

        for key, value in config_data.items():
            if key in class_fields:
                field = class_fields[key]
                try:
                    # Handle type conversion for specific types
                    converted_value = convert_max_config_value(
                        value=value,
                        field_type=field.type,
                        field_name=key,
                    )
                    filtered_config[key] = converted_value
                except Exception as e:
                    raise ValueError(
                        f"Invalid value for field '{key}': {e}"
                    ) from e
            else:
                invalid_keys.append(key)

        if invalid_keys:
            logger.warning(
                f"Ignoring unknown configuration keys for {cls.__name__}: {invalid_keys}"
            )

        # Create and return the config instance
        try:
            return cls(**filtered_config)
        except Exception as e:
            raise ValueError(
                f"Failed to create {cls.__name__} instance: {e}"
            ) from e

    def cli_arg_parsers(
        self,
        choices_provider: dict[str, list[str]] | None = None,
        description: str | None = None,
    ) -> argparse.ArgumentParser:
        """Create an ArgumentParser populated with all MAXConfig fields as arguments.

        This creates a parser with proper add_argument() calls for each field,
        using the loaded config values as defaults. The parser's parse_args() method
        is wrapped to automatically convert parsed string values back to their proper
        types (e.g., enum objects, proper data types) using MAXConfig's type conversion
        logic.

        Args:
            choices_provider: Optional dictionary mapping field names to their valid choices.
                             This allows external code to specify choices for specific fields.
            description: Optional description for the argument parser.

        Usage:
            ```python
            # Basic usage with config file defaults
            config = KVCacheConfig.from_config_file("kv_cache.yaml")
            parser = config.cli_arg_parsers()
            args = parser.parse_args()  # Gets config file values as defaults
            main(args)  # args.cache_strategy will be proper enum object

            # With choices for validation
            choices = {"backend": ["modular", "vllm"], "dataset_name": ["sharegpt", "random"]}
            parser = config.cli_arg_parsers(choices_provider=choices)
            parser.add_argument("--custom-arg", help="Custom argument")
            args = parser.parse_args(["--backend", "vllm"])  # Validates against choices

            # Empty args uses config file defaults
            args = parser.parse_args([])  # All values from config file
            ```

        Returns:
            A configured ArgumentParser with enhanced parse_args() method that:
            - Uses loaded config values as argument defaults
            - Automatically converts parsed values to proper types (enums, etc.)
            - Maintains compatibility with standard argparse usage

        Note:
            The returned parser's parse_args() method automatically handles type
            conversion, so enum fields will return actual enum objects rather than
            strings, matching the behavior of loading from config files.
        """

        # Create parser
        parser = argparse.ArgumentParser(description=description)
        choices_provider = choices_provider or {}

        for field_obj in fields(self):
            # Skip internal fields
            if field_obj.name.startswith("_"):
                continue

            field_name = field_obj.name.replace("_", "-")
            arg_name = f"--{field_name}"
            field_type = field_obj.type

            # Use helper function to determine argparse parameters
            arg_type, action = _get_argparse_type_and_action(
                field_type, field_obj.default
            )

            # Build argument kwargs
            # Use the actual config value, not the field default
            field_value = getattr(self, field_obj.name)

            # Handle enum types specially - argparse expects string values for enums
            if (
                hasattr(field_value, "value")
                and hasattr(field_value, "__class__")
                and issubclass(field_value.__class__, enum.Enum)
            ):
                # For enums, use the string value as default but we'll need to convert back
                arg_kwargs = {
                    "default": field_value.value
                    if field_value
                    else field_obj.default
                }
            else:
                arg_kwargs = {"default": field_value}

            # Apply choices from choices_provider
            if field_obj.name in choices_provider:
                arg_kwargs["choices"] = choices_provider[field_obj.name]

            # Add help from the config class if available
            if hasattr(self, "help"):
                help_dict = self.help()
                if field_obj.name in help_dict:
                    arg_kwargs["help"] = help_dict[field_obj.name]

            # Add argument with appropriate type and action
            if action:
                # Boolean fields with action
                arg_kwargs["action"] = action
                parser.add_argument(arg_name, **arg_kwargs)  # type: ignore
            elif get_origin(field_type) is list:
                # List fields need nargs
                arg_kwargs.update({"type": arg_type, "nargs": "*"})
                parser.add_argument(arg_name, **arg_kwargs)  # type: ignore
            else:
                # Regular typed fields
                arg_kwargs["type"] = arg_type
                parser.add_argument(arg_name, **arg_kwargs)  # type: ignore

        # Create a wrapper class to override parse_args method
        class MAXConfigArgumentParser(argparse.ArgumentParser):
            def __init__(
                self,
                parser_instance: argparse.ArgumentParser,
                config_instance: MAXConfig,
            ):
                # Copy all attributes from the original parser
                self.__dict__.update(parser_instance.__dict__)
                self._config_instance = config_instance
                self._original_parse_args = parser_instance.parse_args

            def parse_args(  # type: ignore[override]
                self,
                args: list[str] | None = None,
                namespace: argparse.Namespace | None = None,
            ):
                # Parse with the original method
                parsed_namespace = self._original_parse_args(args, namespace)

                # Convert string values back to proper types using MAXConfig conversion logic
                converted_dict = {}
                class_fields = {
                    field.name: field for field in fields(self._config_instance)
                }

                for attr_name in dir(parsed_namespace):
                    if attr_name.startswith("_"):
                        continue

                    parsed_value = getattr(parsed_namespace, attr_name)

                    if attr_name in class_fields:
                        field = class_fields[attr_name]
                        try:
                            # Use the same conversion logic as from_config_file
                            converted_value = convert_max_config_value(
                                value=parsed_value,
                                field_type=field.type,
                                field_name=attr_name,
                            )
                            converted_dict[attr_name] = converted_value
                        except Exception:
                            # If conversion fails, keep the parsed value
                            converted_dict[attr_name] = parsed_value
                    else:
                        # Unknown fields remain as-is
                        converted_dict[attr_name] = parsed_value

                return argparse.Namespace(**converted_dict)

        # Return the wrapped parser
        return MAXConfigArgumentParser(parser, self)


# frozen is False (for now) because of _available_cache_memory being set by
# internal code.
@dataclass(frozen=False)
class KVCacheConfig(MAXConfig):
    cache_strategy: KVCacheStrategy = KVCacheStrategy.MODEL_DEFAULT
    """The cache strategy to use. This defaults to :obj:`model_default`, which will set the cache
    strategy based on the default strategy for the architecture requested.

    You can also force the engine to use a specific caching strategy: :obj:`continuous` | :obj:`paged`.
    """

    kv_cache_page_size: int = 128
    """The number of tokens in a single page in the paged KVCache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for the paged attention KVCache."""

    enable_kvcache_swapping_to_host: bool = False
    """Whether to enable swapping the paged attention KVCache blocks to host memory when device blocks are evicted."""

    device_memory_utilization: float = 0.9
    """The fraction of available device memory that the process should consume.

    This is used to inform the size of the KVCache workspace. The calculation is:

    .. math::

        kv\\_cache\\_workspace = (total\\_free\\_memory \\times device\\_memory\\_utilization) - model\\_weights\\_size
    """

    host_kvcache_swap_space_gb: float = 50.0
    """The amount of host memory to use for the host KVCache in GiB.

    This space is only allocated when kvcache_swapping_to_host is enabled.
    """

    _available_cache_memory: int | None = None
    """The amount of available cache memory in bytes. This should only be set by internal code."""

    _config_file_section_name: str = "kv_cache_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "cache_strategy": "Force a specific cache strategy: 'paged' or 'continuous'. If not provided, the optimal caching strategy for the model requested will be selected.",
            "kv_cache_page_size": "The number of tokens in a single page in the paged KVCache. Default is set to 128.",
            "enable_prefix_caching": "Whether to enable prefix caching for the paged attention KVCache. This defaults to false.",
            "enable_kvcache_swapping_to_host": "Whether to enable swapping the paged attention KVCache blocks to host memory when device blocks are evicted. This defaults to false.",
            "device_memory_utilization": "The fraction of available device memory that the process should consume. This is used to inform the size of the KVCache workspace: kv_cache_workspace = (total_free_memory * device_memory_utilization) - model_weights_size. Default is set to 0.9.",
            "host_kvcache_swap_space_gb": "The amount of host memory to use for the host KVCache in GiB. This is only used when kvcache_swapping_to_host is enabled. Default is set to 50.0.",
        }


@dataclass
class SamplingConfig(MAXConfig):
    in_dtype: DType = DType.float32
    """The data type of the input tokens."""

    out_dtype: DType = DType.float32
    """The data type of the output logits."""

    enable_structured_output: bool = False
    """Enable structured generation/guided decoding for the server. This allows the user to pass a json
    schema in the response_format field, which the LLM will adhere to."""

    enable_variable_logits: bool = False
    """Enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with
    their associated logit_offsets. This is needed to produce additional logits for echo and speculative
    decoding purposes."""

    do_penalties: bool = False
    """Whether to apply frequency and presence penalties to the model's output."""

    enable_min_tokens: bool = False
    """Whether to enable min_tokens, which blocks the model from generating
    stopping tokens before the min_tokens count is reached. This defaults to
    false.
    """

    _config_file_section_name: str = "sampling_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "enable_structured_output": "Whether to enable constrained decoding in the text generation pipeline. This defaults to false.",
            "enable_variable_logits": "Whether to enable the sampling graph to accept a ragged tensor of different sequences as inputs, along with their associated logit_offsets. This is needed to produce additional logits for echo and speculative decoding purposes. This defaults to false.",
            "enable_min_tokens": "Whether to enable min_tokens, which blocks the model from generating stopping tokens before the min_tokens count is reached. This defaults to false.",
            "do_penalties": "Whether to apply frequency and presence penalties to the model's output. This defaults to false.",
            "in_dtype": "The data type of the input tokens. This defaults to float32.",
            "out_dtype": "The data type of the output logits. This defaults to float32.",
        }


@dataclass
class ProfilingConfig(MAXConfig):
    gpu_profiling: GPUProfilingMode = GPUProfilingMode.OFF
    """Whether to enable GPU profiling of the model."""

    _config_file_section_name: str = "profiling_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    def __post_init__(self):
        gpu_profiling_env = os.environ.get("MODULAR_ENABLE_PROFILING", "off")

        if self.gpu_profiling == GPUProfilingMode.OFF:
            try:
                self.gpu_profiling = GPUProfilingMode(gpu_profiling_env)
            except ValueError:
                valid_values = [mode.value for mode in GPUProfilingMode]
                raise ValueError(  # noqa: B904
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "gpu_profiling": "Whether to turn on GPU profiling for the model. This defaults to 'off'.",
        }


@dataclass
class LoRAConfig(MAXConfig):
    enable_lora: bool = False
    """Enables LoRA on the server"""

    lora_paths: list[str] = field(default_factory=list)
    """List of statically defined LoRA paths"""

    max_lora_rank: int = 16
    """Maximum rank of all possible LoRAs"""

    max_num_loras: int = 100
    """The maximum number of active LoRAs in a batch"""

    lora_request_endpoint: str = field(default_factory=generate_zmq_ipc_path)
    """The request endpoint for ZMQ communication"""

    lora_response_endpoint: str = field(default_factory=generate_zmq_ipc_path)
    """The response endpoint for ZMQ communication"""

    _config_file_section_name: str = "lora_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @staticmethod
    def help() -> dict[str, str]:
        return {
            "enable_lora": "Enables LoRA on the server",
            "lora_paths": "List of paths to the LoRAs.",
            "max_lora_rank": "The maximum rank of all possible LoRAs. Typically 8 or 16. Default is 16.",
            "max_num_loras": "The maximum number of active LoRAs in a batch. Default is 100.",
            "lora_request_endpoint": "The ZMQ request endpoint for IPC.",
            "lora_response_endpoint": "The ZMQ response endpoint for IPC.",
        }
