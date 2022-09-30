# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
from types import TracebackType

from modular.utils.typing import Optional, Type


class LoggingContext:
    """Context manager to temporarily modify the logging configuration.

    Adapted from: https://docs.python.org/3/howto/logging-cookbook.html.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        level: Optional[int] = None,
        handler: Optional[logging.Handler] = None,
        close: bool = True,
    ) -> None:
        self.logger = logger or logging.getLogger()
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
