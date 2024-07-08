# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

__doc__ = """
Logging utility library

This is a drop-in replacement for the Python standard library's logging module.
Modular developers are strongly encouraged to use this instead of the built-in
logging module, because it overrides the default logger so that we can more
easily separate what's logged by our tools from the often substantial debug logs
of third-party libraries (e.g. TensorFlow).
"""

from logging import *  # type: ignore # noqa: F403
from logging import getLogger as _getLogger
from types import TracebackType
from typing import Any, Optional, Type


def getLogger(name: Optional[str] = None) -> Logger:
    name = name or __name__
    return _getLogger(name)


def info(message: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore
    getLogger().info(f"{message}", *args, **kwargs)


def error(message: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore
    getLogger().error(f"{message}", *args, **kwargs)


def critical(message: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore
    getLogger().critical(f"{message}", *args, **kwargs)


def debug(message: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore
    getLogger().debug(f"{message}", *args, **kwargs)


def warning(message: Any, *args: Any, **kwargs: Any) -> None:  # type: ignore
    getLogger().warning(f"{message}", *args, **kwargs)


class LoggingContext:
    """Context manager to temporarily modify the logging configuration.

    Adapted from: https://docs.python.org/3/howto/logging-cookbook.html.
    """

    def __init__(
        self,
        logger: Optional[Logger] = None,
        level: Optional[int] = None,
        handler: Optional[Handler] = None,
        close: bool = True,
    ) -> None:
        self.logger = logger or getLogger()
        self.level = level
        self.handler = handler
        self.close = close

    def __enter__(self) -> None:
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(
        self,
        exception_type: Optional[Type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()


del Any, Optional, TracebackType, Type
