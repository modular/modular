# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the subprocess package."""


from sys import external_call
from sys.ffi import C_char
from sys.info import os_is_windows

from memory import UnsafePointer

from utils import StringRef


struct _POpenHandle:
    """Handle to an open file descriptor opened via popen."""

    var _handle: UnsafePointer[NoneType]

    fn __init__(inout self, cmd: String, mode: String = "r") raises:
        """Construct the _POpenHandle using the command and mode provided.

        Args:
          cmd: The command to open.
          mode: The mode to open the file in (the mode can be "r" or "w").
        """
        constrained[
            not os_is_windows(), "popen is only available on unix systems"
        ]()

        if mode != "r" and mode != "w":
            raise "the mode specified `" + mode + "` is not valid"

        self._handle = external_call["popen", UnsafePointer[NoneType]](
            cmd.unsafe_ptr(), mode.unsafe_ptr()
        )

        if not self._handle:
            raise "unable to execute the command `" + cmd + "`"

    fn __del__(owned self):
        """Closes the handle opened via popen."""
        _ = external_call["pclose", Int32](self._handle)

    fn read(self) raises -> String:
        """Reads all the data from the handle.

        Returns:
          A string containing the output of running the command.
        """
        var len: Int = 0
        var line = UnsafePointer[C_char]()
        var res = String("")

        while True:
            var read = external_call["getline", Int](
                Reference(line), Reference(len), self._handle
            )
            if read == -1:
                break
            res += StringRef(line, read)

        if line:
            external_call["free", NoneType](line.bitcast[NoneType]())

        return res.rstrip()


fn run(cmd: String) raises -> String:
    """Runs the specified command and returns the output as a string.

    Returns:
      The output of running the command as a string.
    """
    var hdl = _POpenHandle(cmd)
    return hdl.read()
