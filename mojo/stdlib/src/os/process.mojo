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
"""Implements os methods for dealing with processes.

Example:

```mojo
from os import Process
```
"""

from sys import (
    external_call,
    os_is_linux,
    os_is_macos,
    os_is_windows,
)
from sys._libc import vfork, execvp, kill, SignalCodes
from sys.ffi import OpaquePointer, c_char, c_int, c_str_ptr
from sys.os import sep

from memory import UnsafePointer

# ===----------------------------------------------------------------------=== #
# Process execution
# ===----------------------------------------------------------------------=== #


struct Process:
    """Create and manage child processes from file executables.

    Example usage:
    ```
    child_process = Process.run("ls", List[String]("-lha"))
    if child_process.interrupt():
        print("Successfully interrupted.")
    ```
    """

    var child_pid: c_int
    """Child process id."""

    fn __init__(mut self, child_pid: c_int):
        """Struct to manage metadata about child process.
        Use the `run` static method to create new process.

        Args:
          child_pid: The pid of child processed returned by `vfork` that the struct will manage.
        """

        self.child_pid = child_pid

    fn _kill(self, signal: Int) -> Bool:
        # `kill` returns 0 on success and -1 on failure
        return kill(self.child_pid, signal) > -1

    fn hangup(self) -> Bool:
        """Send the Hang up signal to the managed child process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.HUP)

    fn interrupt(self) -> Bool:
        """Send the Interrupt signal to the managed child process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.INT)

    fn kill(self) -> Bool:
        """Send the Kill signal to the managed child process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.KILL)

    @staticmethod
    fn run(path: String, argv: List[String]) raises -> Process:
        """Spawn new process from file executable.

        Args:
          path: The path to the file.
          argv: A list of string arguments to be passed to executable.

        Returns:
          An instance of `Process` struct.
        """

        @parameter
        if os_is_linux() or os_is_macos():
            var file_name = path.split(sep)[-1]
            var pid = vfork()
            if pid == 0:
                var arg_count = len(argv)
                var argv_array_ptr_cstr_ptr = UnsafePointer[c_str_ptr].alloc(
                    arg_count + 2
                )
                var offset = 0
                # Arg 0 in `argv` ptr array should be the file name
                argv_array_ptr_cstr_ptr[offset] = file_name.unsafe_cstr_ptr()
                offset += 1

                for arg in argv:
                    argv_array_ptr_cstr_ptr[offset] = arg[].unsafe_cstr_ptr()
                    offset += 1

                # `argv` ptr array terminates with NULL PTR
                argv_array_ptr_cstr_ptr[offset] = c_str_ptr()

                _ = execvp(path.unsafe_cstr_ptr(), argv_array_ptr_cstr_ptr)

                # This will only get reached if exec call fails to replace currently executing code
                argv_array_ptr_cstr_ptr.free()
                raise Error("Failed to execute " + path)
            elif pid < 0:
                raise Error("Unable to fork parent")

            return Process(child_pid=pid)
        elif os_is_windows():
            constrained[
                False, "Windows process execution currently not implemented"
            ]()
            return abort[Process]()
        else:
            constrained[
                False, "Unknown platform process execution not implemented"
            ]()
            return abort[Process]()
