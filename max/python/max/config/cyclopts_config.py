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
"""Shared utilities for configuring Cyclopts CLI applications.

This module provides reusable functions for setting up Cyclopts applications
with environment variable and YAML config file support, following the pattern
established in benchmark_pipeline_latency.py.

Configuration precedence (highest to lowest):
1. CLI arguments
2. Config files (YAML)
3. Environment variables (MODULAR_*)
4. Defaults in config classes
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from cyclopts import App, Parameter
from cyclopts.config import Env, Yaml


class CycloptsHelpFormatter(str, Enum):
    """Help formatter options for Cyclopts applications."""

    DEFAULT = "default"
    PLAIN = "plain"


def create_cyclopts_app(
    name: str,
    help_text: str,
    help_formatter: CycloptsHelpFormatter = CycloptsHelpFormatter.PLAIN,
) -> App:
    """Create a Cyclopts App with environment variable config source.

    This creates an App configured to load from environment variables with
    the MODULAR_ prefix. The app is configured early so that --help can
    display environment variable options.

    Configuration precedence (highest to lowest):
    1. CLI arguments
    2. Config files (YAML) - if setup_config_file_meta_command is used
    3. Environment variables (MODULAR_*)
    4. Defaults in config classes

    Args:
        name: Name of the CLI application.
        help_text: Help text/description for the application.
        help_formatter: Help formatter style to use.

    Returns:
        A configured Cyclopts App instance with environment variable config source.

    Example:
        ```python
        app = create_cyclopts_app(
            name="my_cli",
            help_text="My CLI application",
        )
        ```
    """
    return App(
        name=name,
        help=help_text,
        help_formatter=help_formatter.value,
        config=[
            # Load from environment variables with MODULAR_ prefix
            # Environment variable names follow pattern: MODULAR_<PARAM_NAME>
            Env(prefix="MODULAR_"),
        ],
    )


def _setup_config_sources(
    app: App,
    config_files: list[Path],
    root_keys: str | None = None,
    must_exist: bool = False,
    search_parents: bool = True,
    tokens: list[str] | None = None,
) -> None:
    """Set up config sources on the app with YAML files overriding env vars.

    This is a helper function used by both the meta command and test utilities
    to ensure consistent config source setup.

    Args:
        app: The Cyclopts App instance to configure.
        config_files: List of YAML config file paths to add as config sources.
        root_keys: Optional root key(s) in YAML files to look for config under.
        must_exist: Whether config files must exist.
        search_parents: Whether to search parent directories for config files.
        tokens: Optional CLI tokens to pass to the app after setting up config sources.
               If provided, app(tokens) will be called after config setup.
    """
    if config_files:
        # Get existing config sources (Env source)
        existing_config = list(app.config) if app.config else []
        config_sources: list[Any] = []

        # Add YAML config files first (higher priority, so they override env vars)
        for config_file_path in config_files:
            if must_exist and not config_file_path.exists():
                raise FileNotFoundError(
                    f"Config file not found: {config_file_path}"
                )
            yaml_config = Yaml(
                config_file_path,
                root_keys=root_keys,
                must_exist=must_exist,
                search_parents=search_parents,
            )
            config_sources.append(yaml_config)

        # Add existing config sources (Env) after YAML (lower priority)
        config_sources.extend(existing_config)

        # Set the config sources on the app (YAML files + env vars)
        app.config = config_sources

    # Invoke the app with tokens if provided
    if tokens is not None:
        app(tokens)


def setup_config_file_meta_command(
    app: App,
    root_keys: str | None = None,
    must_exist: bool = False,
    search_parents: bool = True,
    args: list[str] | None = None,
) -> None:
    """Set up a meta command to handle config file specification.

    This function adds a meta command to the Cyclopts app that allows users
    to specify YAML config files via --config-file arguments. Multiple config
    files can be provided; later files override earlier ones.

    Configuration precedence:
    - CLI args → config files → environment variables → defaults
    - In Cyclopts, config sources are applied in order; the first source takes precedence.
    - To achieve the desired precedence, YAML files are set first (higher priority),
      then env vars are added (lower priority).

    Args:
        app: The Cyclopts App instance to configure.
        root_keys: Optional root key(s) in YAML files to look for config under.
                  If None, config is expected at the root level.
        must_exist: Whether config files must exist. If False, missing files are ignored.
        search_parents: Whether to search parent directories for config files.
        args: Optional list of arguments to pass to app.meta(). If provided, these will be
              used instead of reading from sys.argv. Useful for testing. If None,
              app.meta() will read from sys.argv.

    Example:
        ```python
        app = create_cyclopts_app("my_cli", "My CLI")
        setup_config_file_meta_command(app, root_keys="my_config")
        # Now users can run: my_cli --config-file config.yaml --other-arg value
        ```

    Example for testing:
        ```python
        setup_config_file_meta_command(
            app,
            root_keys="test_config",
            args=["--config-file", "config.yaml", "--other-arg", "value"]
        )
        ```

    Note:
        This function modifies the app in-place and invokes app.meta() at the end
        to handle config file setup before processing the main command. If args
        is provided, those arguments will be used instead of reading from sys.argv.
    """

    @app.meta.default
    def meta(
        *tokens: Annotated[
            str, Parameter(show=False, allow_leading_hyphen=True)
        ],
        config_file: Annotated[
            Path | None,
            Parameter(consume_multiple=True),
        ] = None,
    ) -> None:
        """Meta command to configure config file sources.

        Args:
            *tokens: Remaining CLI tokens to pass through to the main command.
            config_file: Path(s) to YAML config file(s). Can be specified multiple times.
                        Multiple config files can be provided; later files override earlier ones.
        """
        # Build list of config files
        # Normalize config_file to always be a list (consume_multiple may return single value or list)
        if config_file is None:
            config_files = []
        elif isinstance(config_file, list):
            config_files = config_file
        else:
            config_files = [config_file]

        _setup_config_sources(
            app=app,
            config_files=config_files,
            root_keys=root_keys,
            must_exist=must_exist,
            search_parents=search_parents,
            tokens=list(tokens),
        )

    # Use user provided args than reading from sys.argv
    if args is not None:
        # Extract all --config-file arguments and collapse remaining args
        config_files: list[Path] = []
        remaining_tokens: list[str] = []

        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "--config-file" and i + 1 < len(args):
                # Format: --config-file path
                config_file_path = Path(args[i + 1]).resolve()
                config_files.append(config_file_path)
                i += 2  # Skip both --config-file and its value
            elif arg.startswith("--config-file="):
                # Format: --config-file=path
                config_file_path = Path(arg.split("=", 1)[1]).resolve()
                config_files.append(config_file_path)
                i += 1
            else:
                # Keep this argument as a remaining token
                remaining_tokens.append(arg)
                i += 1

        _setup_config_sources(
            app=app,
            config_files=config_files,
            root_keys=root_keys,
            must_exist=must_exist,
            search_parents=search_parents,
            tokens=remaining_tokens,
        )
    else:
        app.meta()
