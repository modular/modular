# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tools for reading metadata from a checkpoint file without completely loading
all tensors.
"""

import os
from collections import InlineArray
from os import PathLike
from pathlib import Path
from sys import sizeof

from max.tensor import Tensor

# 0x93 🔥 + + 0x93
alias _SERIALIZATION_HEADER = InlineArray[Int8, 8](
    0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x2B, 0x2B, 0x93
)

# Serialization constants
alias _SERIALIZATION_MAJOR_FORMAT: UInt32 = 0
alias _SERIALIZATION_MINOR_FORMAT: UInt32 = 1


@value
struct VersionInfo:
    """Struct containing major and minor version values.

    The MAX checkpoint format guarantees backwards compatibility, meaning that
    older checkpoints can always be read by newer versions of the library.

    Forward compatibility is guaranteed across the same major version. If the
    major version is bumped up, it means that there has been a backwards-
    incompatible change made.
    """

    var major_version: UInt32
    var minor_version: UInt32

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.major_version, ".", self.minor_version)


def current_version() -> VersionInfo:
    return VersionInfo(_SERIALIZATION_MAJOR_FORMAT, _SERIALIZATION_MINOR_FORMAT)


def read_version[PathLike: PathLike](path: PathLike) -> VersionInfo:
    """Reads the header of a MAX checkpoint and returns the version info.

    Args:
        path: Path to MAX checkpoint file.

    Returns:
        `VersionInfo` containing the major and minor versions of the file.

    Raises:
        Error if the serialization header does not match.
    """
    with open(path, "rb") as f:
        return _read_version(path, f)


def _read_version[
    PathLike: PathLike
](path: PathLike, f: FileHandle) -> VersionInfo:
    """Reads the header of a MAX checkpoint and returns the version info.

    Args:
        path: `Path` to MAX checkpoint file. Only used for error reporting
            purposes, since the FileHandle is provided.
        f: Handle for reading MAX checkpoint file.

    Returns:
        `VersionInfo` containing the major and minor versions of the file.

    Raises:
        Error if the serialization header does not match.
    """
    var header_buf = _serialization_header()
    var header = f.read_bytes(8)
    if len(header) != 8:
        raise "Invalid checkpoint file: " + path.__fspath__()
    for i in range(8):
        if header[i] != header_buf[i]:
            raise "Given file is not a max.graph checkpoint."

    var major_version = _read_int[DType.uint32](f)
    var minor_version = _read_int[DType.uint32](f)
    return VersionInfo(major_version, minor_version)


@always_inline
fn _read_int[type: DType](f: FileHandle) raises -> Scalar[type]:
    """Reads an int value from a file."""
    var size = sizeof[type]()
    var bytes_tensor = Tensor[DType.uint8](f.read_bytes(size))
    return bytes_tensor.unsafe_ptr().bitcast[Scalar[type]]().load()


def _serialization_header() -> List[UInt8]:
    # TODO: Using _SERIALIZATION_HEADER raises errors.
    return List[UInt8](0x93, 0xF0, 0x9F, 0x94, 0xA5, 0x2B, 0x2B, 0x93)
