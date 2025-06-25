# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from pathlib import Path

from _pytest.logging import LogCaptureFixture
from modular.utils import logging


def test_LoggingContext_logger(caplog: LogCaptureFixture) -> None:
    logging.getLogger().setLevel(logging.DEBUG)

    levels = ["debug", "info", "warning", "error", "critical"]
    for level in levels:
        getattr(logging, level)(f"Some {level}")

    assert len(caplog.records) == 5
    for rec, level in zip(caplog.records, levels):
        assert rec.msg == f"Some {level}"
        assert rec.levelname == level.upper()
        assert rec.name == "modular.utils.logging"
    assert True


def test_LoggingContext_level(caplog: LogCaptureFixture) -> None:
    logging.getLogger().setLevel(logging.INFO)

    logging.info("Some info")
    logging.debug("Some debug")
    with logging.LoggingContext(level=logging.DEBUG):
        logging.debug("Some other debug")

    assert len(caplog.records) == 2
    assert caplog.records[0].msg == "Some info"
    assert caplog.records[1].msg == "Some other debug"


def test_LoggingContext_handler(
    caplog: LogCaptureFixture, tmp_path: Path
) -> None:
    log_file = tmp_path / "log.txt"
    log_file.unlink(missing_ok=True)

    logging.getLogger().setLevel(logging.INFO)
    with logging.LoggingContext(handler=logging.FileHandler(log_file)):
        logging.info("Some info")
    logging.info("Some other info")

    assert len(caplog.records) == 2
    assert caplog.records[0].msg == "Some info"
    assert caplog.records[1].msg == "Some other info"
    assert log_file.read_text() == "Some info\n"
