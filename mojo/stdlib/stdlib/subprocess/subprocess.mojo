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
"""Implements the subprocess package."""


import sys._libc as libc
from sys import external_call
from sys._libc import FILE_ptr, pclose, popen
from sys.ffi import c_char
from sys.info import CompilationTarget

from memory import Span


struct _POpenHandle:
    """Handle to an open file descriptor opened via popen."""

    var _handle: FILE_ptr

    fn __init__(out self, var cmd: String, var mode: String = "r") raises:
        """Construct the _POpenHandle using the command and mode provided.

        Args:
          cmd: The command to open.
          mode: The mode to open the file in (the mode can be "r" or "w").
        """
        constrained[
            not CompilationTarget.is_windows(),
            "popen is only available on unix systems",
        ]()

        if mode != "r" and mode != "w":
            raise Error("the mode specified `", mode, "` is not valid")

        self._handle = popen(cmd.unsafe_cstr_ptr(), mode.unsafe_cstr_ptr())

        if not self._handle:
            raise Error("unable to execute the command `", cmd, "`")

    fn __del__(deinit self):
        """Closes the handle opened via popen."""
        _ = pclose(self._handle)

    fn read(self) raises -> String:
        """Reads all the data from the handle.

        Returns:
            A string containing the output of running the command.

        Raises:
            This method raises if:
            * There is an IO error reading from the subprocess.
            * The data written by the subprocess is not valid UTF-8.
        """
        var len: Int = 0
        var line = UnsafePointer[c_char]()
        var res = String()

        while True:
            var read = external_call["getline", Int](
                Pointer(to=line), Pointer(to=len), self._handle
            )
            if read == -1:
                break

            # Note: This will raise if the subprocess yields non-UTF-8 bytes.
            res += StringSlice(
                from_utf8=Span(ptr=line.bitcast[Byte](), length=UInt(read))
            )

        if line:
            libc.free(line.bitcast[NoneType]())

        return String(res.rstrip())


fn run(cmd: String) raises -> String:
    """Runs the specified command and returns the output as a string.

    This function executes the given command in a subprocess, captures its
    standard output, and returns it as a string. It automatically handles
    opening and closing the subprocess.

    Args:
        cmd: The command to execute as a string.

    Returns:
        The standard output of the command as a string, with trailing
        whitespace removed.

    Raises:
        This function raises if:
        * The command cannot be executed.
        * There is an IO error reading from the subprocess.
        * The data written by the subprocess is not valid UTF-8.
    """
    var hdl = _POpenHandle(cmd)
    return hdl.read()
