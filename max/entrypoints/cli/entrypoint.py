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

"""Entrypoint for generalized click cli."""

from __future__ import annotations

import logging

import click


def configure_cli_logging(level: str = "INFO") -> None:
    """Configure logging for CLI operations without using Settings.

    Args:
        level: Log level as string (DEBUG, INFO, WARNING, ERROR)
        quiet: If True, set logging to WARNING level
        verbose: If True, set logging to DEBUG level
    """
    if level == "INFO":
        log_level = logging.INFO
    elif level == "DEBUG":
        log_level = logging.DEBUG
    elif level == "WARNING":
        log_level = logging.WARNING
    elif level == "ERROR":
        log_level = logging.ERROR
    else:
        raise ValueError(f"Unsupported log level: {level}")

    # Clear existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Set up log filtering for MAX components
    components_to_log = [
        "root",
        "max.entrypoints",
        "max.pipelines",
        "max.serve",
    ]

    def log_filter(record: logging.LogRecord) -> bool:
        """Filter to only show logs from MAX components."""
        return any(
            record.name == component or record.name.startswith(component + ".")
            for component in components_to_log
        )

    console_handler.addFilter(log_filter)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sse_starlette.sse").setLevel(
        max(log_level, logging.INFO)
    )


class ModelGroup(click.Group):
    def get_command(
        self, ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\n"
            f"Supported commands: {supported}"
        )
