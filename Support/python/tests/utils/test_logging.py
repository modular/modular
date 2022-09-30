# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import logging
from pathlib import Path

from _pytest.logging import LogCaptureFixture

from modular.utils.logging import LoggingContext


def test_LoggingContext_level(caplog: LogCaptureFixture):
    logging.getLogger().setLevel(logging.INFO)

    logging.info("Some info")
    logging.debug("Some debug")
    with LoggingContext(level=logging.DEBUG):
        logging.debug("Some other debug")

    assert len(caplog.records) == 2
    assert caplog.records[0].msg == "Some info"
    assert caplog.records[1].msg == "Some other debug"


def test_LoggingContext_handler(caplog: LogCaptureFixture):
    log_file = Path(__file__).parent / ".artifacts" / "log.txt"
    log_file.unlink(missing_ok=True)

    logging.getLogger().setLevel(logging.INFO)
    with LoggingContext(handler=logging.FileHandler(log_file)):
        logging.info("Some info")
    logging.info("Some other info")

    assert len(caplog.records) == 2
    assert caplog.records[0].msg == "Some info"
    assert caplog.records[1].msg == "Some other info"
    assert log_file.read_text() == "Some info\n"
