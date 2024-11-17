# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from memory import UnsafePointer
from tempfile import NamedTemporaryFile
from sys import sizeof

import testing

from max.graph.checkpoint.metadata import (
    read_version,
    _serialization_header,
    VersionInfo,
)


@always_inline
fn _write_int[type: Intable](ref object: type, f: FileHandle) raises:
    """Writes an int value to a file."""
    var ptr = UnsafePointer.address_of(object).bitcast[UInt8]()
    f._write(ptr, sizeof[type]())


def test_read_version():
    # Write header and metadata
    var header_buf = _serialization_header()
    var header = String(header_buf)
    var major_version: UInt32 = 1234567
    var minor_version: UInt32 = 8910
    with NamedTemporaryFile(name=str("test_simple")) as TEMP_FILE:
        with open(TEMP_FILE.name, "wb") as f:
            f.write(header)
            _write_int(major_version, f)
            _write_int(minor_version, f)
            # Write some arbitary text
            f.write(str("this should not be read "))

        version = read_version(TEMP_FILE.name)
        testing.assert_equal(1234567, version.major_version)
        testing.assert_equal(8910, version.minor_version)


def test_bad_header():
    # Write an invalid header (missing characters)
    var header_buf = List[UInt8](0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x2B, 0x0)
    var header = String(header_buf)
    var major_version: UInt32 = 1234567
    var minor_version: UInt32 = 8910
    with NamedTemporaryFile(name=str("test_simple")) as TEMP_FILE:
        with open(TEMP_FILE.name, "wb") as f:
            f.write(header)
            _write_int(major_version, f)
            _write_int(minor_version, f)
            # Write some arbitary text
            f.write(str("this should not be read "))

        with testing.assert_raises(contains="not a max.graph checkpoint"):
            _ = read_version(TEMP_FILE.name)


def main():
    test_read_version()
    test_bad_header()
