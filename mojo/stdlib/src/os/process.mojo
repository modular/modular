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
from collections import List, Optional
from collections.string import StringSlice

from sys import (
    os_is_linux,
    os_is_macos,
    os_is_windows,
)
from sys._libc import (
    vfork,
    execvp,
    exit,
    kill,
    SignalCodes,
    pipe,
    fcntl,
    FcntlCommands,
    FcntlFDFlags,
    close,
)
from sys.ffi import c_char, c_int
from sys.os import sep

from memory import Span, UnsafePointer


# ===----------------------------------------------------------------------=== #
# Process comm.
# ===----------------------------------------------------------------------=== #
struct Pipe:
    """Create a pipe for interprocess communication.

    Example usage:
    ```
    pipe().write_bytes("TEST".as_bytes())
    ```
    """

    var fd_in: Optional[FileDescriptor]
    """File descriptor for pipe input."""
    var fd_out: Optional[FileDescriptor]
    """File descriptor for pipe output."""

    fn __init__(
        mut self,
        in_close_on_exec: Bool = False,
        out_close_on_exec: Bool = False,
    ) raises:
        """Struct to manage interprocess pipe comms.

        Args:
            in_close_on_exec: Close the read side of pipe if `exec` sys. call is issued in process.
            out_close_on_exec: Close the write side of pipe if `exec` sys. call is issued in process.
        """
        var pipe_fds = UnsafePointer[c_int].alloc(2)
        if pipe(pipe_fds) < 0:
            pipe_fds.free()
            raise Error("Failed to create pipe")

        if in_close_on_exec:
            if not self._set_close_on_exec(pipe_fds[0]):
                pipe_fds.free()
                raise Error("Failed to configure input pipe close on exec")

        if out_close_on_exec:
            if not self._set_close_on_exec(pipe_fds[1]):
                pipe_fds.free()
                raise Error("Failed to configure output pipe close on exec")

        self.fd_in = FileDescriptor(Int(pipe_fds[0]))
        self.fd_out = FileDescriptor(Int(pipe_fds[1]))
        pipe_fds.free()

    fn __del__(owned self):
        """Ensures pipes input and output file descriptors are closed, when the object is destroyed.
        """
        self.set_input_only()
        self.set_output_only()

    @staticmethod
    fn _set_close_on_exec(fd: c_int) -> Bool:
        return (
            fcntl(
                fd,
                FcntlCommands.F_SETFD,
                fcntl(fd, FcntlCommands.F_GETFD, 0) | FcntlFDFlags.FD_CLOEXEC,
            )
            == 0
        )

    @always_inline
    fn set_input_only(mut self):
        """Close the output descriptor/ channel for this side of the pipe."""
        if self.fd_out:
            _ = close(rebind[Int](self.fd_out.value()))
            self.fd_out = None

    @always_inline
    fn set_output_only(mut self):
        """Close the input descriptor/ channel for this side of the pipe."""
        if self.fd_in:
            _ = close(rebind[Int](self.fd_in.value()))
            self.fd_in = None

    @always_inline
    fn write_bytes(mut self, bytes: Span[Byte, _]) raises:
        """
        Write a span of bytes to the pipe.

        Args:
            bytes: The byte span to write to this pipe.

        """
        if self.fd_out:
            self.fd_out.value().write_bytes(bytes)
        else:
            raise Error("Can not write from read only side of pipe")

    @always_inline
    fn read_bytes(mut self, mut buffer: Span[mut=True, Byte]) raises -> UInt:
        """
        Read a number of bytes from this pipe.

        Args:
            buffer: Span[Byte] of length n where to store read bytes. n = number of bytes to read.

        Returns:
            Actual number of bytes read.
        """
        if self.fd_in:
            return self.fd_in.value().read_bytes(buffer)

        raise Error("Can not read from write only side of pipe")


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
            var pipe = Pipe(out_close_on_exec=True)
            var exec_err_code = String("EXEC_ERR")

            var pid = vfork()

            if pid == 0:
                """Child process."""
                pipe.set_output_only()

                var arg_count = len(argv)
                var argv_array_ptr_cstr_ptr = UnsafePointer[
                    UnsafePointer[c_char]
                ].alloc(arg_count + 2)
                var offset = 0
                # Arg 0 in `argv` ptr array should be the file name
                argv_array_ptr_cstr_ptr[offset] = file_name.unsafe_cstr_ptr()
                offset += 1

                for arg in argv:
                    argv_array_ptr_cstr_ptr[offset] = arg[].unsafe_cstr_ptr()
                    offset += 1

                # `argv` ptr array terminates with NULL PTR
                argv_array_ptr_cstr_ptr[offset] = UnsafePointer[c_char]()

                _ = execvp(path.unsafe_cstr_ptr(), argv_array_ptr_cstr_ptr)

                # This will only get reached if exec call fails to replace currently executing code
                argv_array_ptr_cstr_ptr.free()

                # Canonical fork/ exec error handling pattern of using a pipe that closes on exec is
                # used to signal error to parent process `https://cr.yp.to/docs/selfpipe.html`
                pipe.write_bytes(exec_err_code.as_bytes())

                exit(1)

            elif pid < 0:
                raise Error("Unable to fork parent")

            pipe.set_input_only()
            var err: Optional[StringSlice[MutableAnyOrigin]] = None
            try:
                var err_len = exec_err_code.byte_length()
                var buf = Span[Byte, MutableAnyOrigin](
                    ptr=UnsafePointer[Byte].alloc(err_len), length=err_len
                )
                buf[0] = 0  # Explicitly default to empty C string
                var bytes_read = pipe.read_bytes(buf)
                err = StringSlice(unsafe_from_utf8=buf)
            except e:
                err = None

            if err and len(err.value()) > 0 and err.value() == exec_err_code:
                raise Error("Failed to execute " + path)

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
