# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from logging import *  # noqa: F403
from logging import getLogger as _getLogger
from types import TracebackType

from modular.utils.typing import Any, Optional, Type


def getLogger(name: Optional[str] = None) -> Logger:
    name = name or __name__
    return _getLogger(name)


def info(message: Any, *args: Any, **kwargs: Any) -> None:
    getLogger().info(f"{message}", *args, **kwargs)


def error(message: Any, *args: Any, **kwargs: Any) -> None:
    getLogger().error(f"{message}", *args, **kwargs)


def critical(message: Any, *args: Any, **kwargs: Any) -> None:
    getLogger().critical(f"{message}", *args, **kwargs)


def debug(message: Any, *args: Any, **kwargs: Any) -> None:
    getLogger().debug(f"{message}", *args, **kwargs)


def warning(message: Any, *args: Any, **kwargs: Any) -> None:
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
