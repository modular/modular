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
from collections import List
_ = Process.run("echo", ["== TEST_ECHO"])
```
"""
from collections import Dict, List, Optional
from collections.string import StringSlice

from memory import LegacyUnsafePointer

from sys import CompilationTarget
from sys._libc import (
    waitpid,
    posix_spawnp,
    posix_spawn_file_actions_t,
    posix_spawn_file_actions_t_ptr,
    posix_spawn_file_actions_init,
    posix_spawn_file_actions_destroy,
    posix_spawn_file_actions_adddup2,
    kill,
    SignalCodes,
    pipe,
    fcntl,
    FcntlCommands,
    FcntlFDFlags,
    close,
    WaitFlags,
)
from sys.ffi import c_char, c_int, c_pid_t, get_errno
from sys.os import abort, sep
from io import FileDescriptor


# ===----------------------------------------------------------------------=== #
# Process comm.
# ===----------------------------------------------------------------------=== #


struct ProcessStatus(Copyable, ImplicitlyCopyable, Movable):
    """Represents the termination status of a process.

    This struct is returned by `poll()` and `wait_process_status()`.
    """

    var exit_code: Optional[Int]
    """The exit code if the process terminated normally."""

    var term_signal: Optional[Int]
    """The signal number that terminated the process."""

    fn __init__(
        out self,
        exit_code: Optional[Int] = None,
        term_signal: Optional[Int] = None,
    ):
        """Initializes a new `ProcessStatus`.

        Args:
            exit_code: The exit code if the process terminated normally.
            term_signal: The signal number that terminated the process.
        """
        self.exit_code = exit_code
        self.term_signal = term_signal

    @staticmethod
    fn running() -> Self:
        """Creates a status for a running process.

        Returns:
            A `ProcessStatus` for a running process.
        """
        return Self()

    fn has_exited(self) -> Bool:
        """Checks if the process has terminated.

        Returns:
            True if the process has terminated, either normally or by a signal.
        """
        return Bool(self.exit_code) or Bool(self.term_signal)


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
        out self,
        in_close_on_exec: Bool = False,
        out_close_on_exec: Bool = False,
    ) raises:
        """Initializes a new `Pipe`.

        Args:
            in_close_on_exec: Close the read side of pipe if an `exec` syscall
              is issued in the process.
            out_close_on_exec: Close the write side of pipe if an `exec`
              syscall is issued in the process.

        Raises:
            Error: If the pipe could not be created or configured.
        """
        var pipe_fds = alloc[c_int](2)
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

    fn __del__(deinit self):
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
        """Writes a span of bytes to the pipe.

        Args:
            bytes: The byte span to write to this pipe.

        Raises:
            Error: If called on a read-only pipe.
        """
        if self.fd_out:
            self.fd_out.value().write_bytes(bytes)
        else:
            raise Error("Can not write from read only side of pipe")

    @always_inline
    fn read_bytes(mut self, mut buffer: Span[mut=True, Byte]) raises -> UInt:
        """Read a number of bytes from this pipe.

        Args:
            buffer: Span[Byte] of length n where to store read bytes. n = number of bytes to read.

        Returns:
            Actual number of bytes read.

        Raises:
            Error: If the pipe is in write-only mode.
        """
        if self.fd_in:
            return self.fd_in.value().read_bytes(buffer)

        raise Error("Can not read from write only side of pipe")


# ===----------------------------------------------------------------------=== #
# Process execution
# ===----------------------------------------------------------------------=== #
struct Process:
    """Create and manage sub processes from file executables.

    The sub process will be automatically "closed out" with `Process.wait` when the Process
    instance is destroyed and `Process.__del__` is called.

    User can manually check status in non-blocking way using `Process.poll` or block the
    caller process with `Process.wait` manually at their chosen code point.

    Note that sub process status is cached on the Process object as such it is NOT thread-safe
    and interacting with a Process instance across multiple threads can lead to race conditions and
    undefined behaviors.

    Example usage:
    ```
    sub_process = Process.run("ls", ["-lha"])
    if sub_process.interrupt():
        print("Successfully interrupted.")
    ```
    """

    var sub_pid: c_pid_t
    """Sub process id."""

    var status: Optional[ProcessStatus]
    """Cached status of the process. `None` if the process has not been waited on yet."""

    fn __init__(out self, sub_pid: c_pid_t):
        """Struct to manage metadata about sub_process.
        Use the `run` static method to create new process.

        Args:
          sub_pid: The pid of sub processed returned by `posix_spawnp` that the struct will manage.
        """

        self.sub_pid = sub_pid
        self.status = None

    fn __del__(deinit self):
        """Waits for the process to exit when the `Process` object is destroyed.
        """
        try:
            _ = self.wait()
        except:
            # Errors in __del__ should be suppressed.
            pass

    fn _kill(mut self, signal: Int) -> Bool:
        try:
            if self.poll().has_exited():
                # Process has already exited, no signal can be sent.
                # This is consistent with `kill` failing with ESRCH.
                return False
        except:
            # If poll fails, fall through and attempt to kill anyway.
            pass

        # `kill` returns 0 on success and -1 on failure
        return kill(self.sub_pid, signal) > -1

    fn _check_status(self, pid: c_pid_t, status: c_int) raises -> ProcessStatus:
        """Helper to decode the result of a waitpid call.

        The decoding logic is a direct implementation of the standard C macros
        used to interpret the `status` integer returned by `waitpid`. These
        macros are defined in `<sys/wait.h>` on POSIX systems.

        This implementation is based on the definitions found in `musl` libc.:
        https://git.musl-libc.org/cgit/musl/tree/include/sys/wait.h

        The core logic relies on the following macro definitions:
        - `#define WEXITSTATUS(s) (((s) & 0xff00) >> 8)`
        - `#define WTERMSIG(s)    ((s) & 0x7f)`
        - `#define WIFEXITED(s)   (WTERMSIG(s) == 0)`

        Note on Endianness:
        This logic is endianness-independent. The `waitpid` status is an integer
        value provided by the kernel. All bitwise operations (`&`, `>>`) are
        performed on this integer's numerical value, not its byte representation
        in memory. The result is therefore consistent across architectures.
        """
        if pid == self.sub_pid:
            # Process has terminated. Decode the status.
            if (status & 0x7F) == 0:
                # Process exited normally. Extract the exit code.
                var code = (status & 0xFF00) >> 8
                return ProcessStatus(exit_code=Optional(Int(code)))
            else:
                # Process was terminated by a signal. Extract the signal number.
                var signal = status & 0x7F
                return ProcessStatus(term_signal=Optional(Int(signal)))
        elif pid == 0:
            # Process is still running (only for non-blocking calls).
            return ProcessStatus.running()
        else:
            # An error occurred.
            var err = get_errno()
            raise Error("waitpid failed with errno " + String(err))

    fn hangup(mut self) -> Bool:
        """Send the Hang up signal to the managed sub process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.HUP)

    fn interrupt(mut self) -> Bool:
        """Send the Interrupt signal to the managed sub process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.INT)

    fn kill(mut self) -> Bool:
        """Send the Kill signal to the managed sub process.

        Returns:
          Upon successful completion, True is returned else False.
        """
        return self._kill(SignalCodes.KILL)

    fn poll(mut self) raises -> ProcessStatus:
        """Check if the sub process has terminated in a non-blocking way.

        This method updates the internal state of the `Process` object.
        If the process has terminated, the status is cached.


        Returns:
            A `ProcessStatus` indicating the status of the process.

            If called multiple times the return value will be the cached status,
            as status should only be retrieved once from the OS.

        Raises:
            Error: If `waitpid` fails.
        """
        if self.status:
            return self.status.value()

        var status: c_int = 0
        var pid = waitpid(
            self.sub_pid, UnsafePointer(to=status), WaitFlags.WNOHANG
        )
        var result = self._check_status(pid, status)
        if result.has_exited():
            self.status = result
        return result

    fn wait(mut self) raises -> ProcessStatus:
        """Wait for the sub process to terminate (blocking).

        This method updates the internal state of the `Process` object.
        If the process has terminated, the status is cached.

        Returns:
          A `ProcessStatus` indicating the process has exited and its status.

          If called multiple times the return value will be the cached status,
          as status should only be retrieved once from the OS.

        Raises:
            Error: If `waitpid` fails or the process does not exit.
        """
        if self.status:
            return self.status.value()

        var status: c_int = 0
        var pid = waitpid(self.sub_pid, UnsafePointer(to=status), 0)
        var result = self._check_status(pid, status)
        if result.has_exited():
            self.status = result
        else:
            # This should not be reachable with a blocking waitpid call.
            raise Error("Blocking waitpid returned without process exiting.")
        return result

    @staticmethod
    fn run(
        var path: String,
        argv: List[String],
        env: Optional[Dict[String, String]] = None,
        stdin: Optional[FileDescriptor] = None,
        stdout: Optional[FileDescriptor] = None,
        stderr: Optional[FileDescriptor] = None,
    ) raises -> Process:
        """Spawn new process from file executable.

        Args:
          path: The path to the file.
          argv: A list of string arguments to be passed to executable.
          env: An optional dictionary of environment variables to be passed to the subprocess.
               If None, the child process inherits the environment of the calling process.
          stdin: An optional file descriptor to be used as the subprocess's standard input.
          stdout: An optional file descriptor to be used as the subprocess's standard output.
          stderr: An optional file descriptor to be used as the subprocess's standard error.

        Returns:
          An instance of `Process` struct.

        Raises:
            Error: If the process fails to spawn.
        """
        # TODO: Add support for StringSlice, StringLiteral run args

        @parameter
        if CompilationTarget.is_linux() or CompilationTarget.is_macos():
            var file_name = String(path.split(sep)[-1])

            var arg_count = len(argv)
            var argv_array_ptr_cstr_ptr = alloc[
                UnsafePointer[mut=False, c_char, ImmutAnyOrigin]
            ](arg_count + 2)
            var offset = 0
            # Arg 0 in `argv` ptr array should be the file name
            argv_array_ptr_cstr_ptr[offset] = file_name.unsafe_cstr_ptr()
            offset += 1

            for var arg in argv:
                argv_array_ptr_cstr_ptr[offset] = arg.unsafe_cstr_ptr()
                offset += 1

            # `argv` ptr array terminates with NULL PTR
            argv_array_ptr_cstr_ptr[offset] = UnsafePointer[
                mut=False, c_char, ImmutAnyOrigin
            ]()
            var path_cptr = path.unsafe_cstr_ptr()

            var pid: c_pid_t = 0
            var envp_ptr = UnsafePointer[
                UnsafePointer[mut=False, c_char, ImmutAnyOrigin], MutAnyOrigin
            ]()
            var env_strs = List[String]()

            if env:
                ref env_dict = env.value()
                var env_count = len(env_dict)
                envp_ptr = alloc[
                    UnsafePointer[mut=False, c_char, ImmutAnyOrigin]
                ](env_count + 1)

                for item in env_dict.items():
                    env_strs.append(item.key + "=" + item.value)

                offset = 0
                for ref env_str in env_strs:
                    envp_ptr[offset] = env_str.unsafe_cstr_ptr()
                    offset += 1

                envp_ptr[env_count] = UnsafePointer[
                    mut=False, c_char, ImmutAnyOrigin
                ]()

            var file_actions = posix_spawn_file_actions_t()
            var file_actions_ptr = UnsafePointer(to=file_actions)
            var use_file_actions = stdin or stdout or stderr

            if use_file_actions:
                if posix_spawn_file_actions_init(file_actions_ptr) != 0:
                    raise Error("Failed to initialize file actions")

            try:
                if use_file_actions:
                    if stdin:
                        if (
                            posix_spawn_file_actions_adddup2(
                                file_actions_ptr, stdin.value().value, 0
                            )
                            != 0
                        ):
                            raise Error("Failed to dup stdin")
                    if stdout:
                        if (
                            posix_spawn_file_actions_adddup2(
                                file_actions_ptr, stdout.value().value, 1
                            )
                            != 0
                        ):
                            raise Error("Failed to dup stdout")
                    if stderr:
                        if (
                            posix_spawn_file_actions_adddup2(
                                file_actions_ptr, stderr.value().value, 2
                            )
                            != 0
                        ):
                            raise Error("Failed to dup stderr")

                var has_error_code = posix_spawnp(
                    UnsafePointer(to=pid),
                    path_cptr,
                    file_actions_ptr if use_file_actions else posix_spawn_file_actions_t_ptr[
                        mut=False, origin=ImmutAnyOrigin
                    ](),
                    argv_array_ptr_cstr_ptr,
                    envp_ptr,
                )

                if has_error_code > 0:
                    raise Error(
                        "Failed to execute "
                        + path
                        + ", EINT error code: "
                        + String(has_error_code)
                    )
            finally:
                if use_file_actions:
                    if posix_spawn_file_actions_destroy(file_actions_ptr) != 0:
                        print("Warning: Failed to destroy file actions.")

            argv_array_ptr_cstr_ptr.free()
            if env:
                envp_ptr.free()

            return Process(sub_pid=pid)
        else:
            constrained[
                False, "Unknown platform process execution not implemented"
            ]()
            abort[prefix="ERROR:"](
                "Unknown platform process execution not implemented"
            )
