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

from collections import List, Optional
from os import Process, Pipe
from sys._libc import SignalCodes
from time import sleep
from memory import Span
from testing import assert_equal


fn long_running_process_wait() raises:
    print("== Test: Wait for a long-running process ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("1")

    print("Running 'sleep 1'...")
    var process = Process.run(command, arguments)

    print("Waiting for process to finish...")
    var status = process.wait()

    if status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())
    elif status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())


fn long_running_process_poll() raises:
    print("== Test: Poll a long-running process ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("2")

    print("Running 'sleep 2'...")
    var process = Process.run(command, arguments)

    print("Polling process status...")
    while not process.poll().has_exited():
        print("Process is still running...")
        sleep(0.5)

    var status = process.status.value()
    print("Process has exited.")
    if status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())
    elif status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())


fn interrupt_process() raises:
    print("== Test: Interrupt a process (SIGINT) ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("5")

    print("Running 'sleep 5'...")
    var process = Process.run(command, arguments)

    print("Sleeping for 1 second before sending SIGINT...")
    sleep(1.0)

    if process.interrupt():
        print("Successfully sent SIGINT.")
    else:
        print("Failed to send SIGINT.")

    var status = process.wait()

    if status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())
        if status.term_signal.value() == SignalCodes.INT:
            print("Termination signal was SIGINT, as expected.")
    elif status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())


fn kill_process() raises:
    print("== Test: Kill a process (SIGKILL) ==")

    var command = "sleep"
    var arguments = List[String]()
    arguments.append("5")

    print("Running 'sleep 5'...")
    var process = Process.run(command, arguments)

    print("Sleeping for 1 second before sending SIGKILL...")
    sleep(1.0)

    if process.kill():
        print("Successfully sent SIGKILL.")
    else:
        print("Failed to send SIGKILL.")

    var status = process.wait()

    if status.term_signal:
        print("Process terminated by signal:", status.term_signal.value())
        if status.term_signal.value() == SignalCodes.KILL:
            print("Termination signal was SIGKILL, as expected.")
    elif status.exit_code:
        print("Process finished with exit code:", status.exit_code.value())


fn pipe_io_redirect() raises:
    print("== Example: Pipe I/O Redirection ==")
    var p_in = Pipe()
    var p_out = Pipe()

    var input_string = "Hello from parent!\n"
    print("Parent writing to child's stdin:", input_string.strip())
    p_in.write_bytes(input_string.as_bytes())
    p_in.set_input_only()  # Close the write end of p_in in parent

    print("Running 'cat' with redirected stdin/stdout...")
    var process = Process.run("cat", [], stdin=p_in.fd_in, stdout=p_out.fd_out)

    var status = process.wait()
    print("Child process finished with exit code:", status.exit_code.value())

    var buf_size = 32  # Buffer for reading child's stdout
    var buf_ptr = alloc[UInt8](buf_size)
    var buf_span = Span[mut=True, UInt8](ptr=buf_ptr, length=buf_size)
    var bytes_read = p_out.read_bytes(buf_span)
    var result_span = buf_span.unsafe_subspan(offset=0, length=Int(bytes_read))
    var result = String(bytes=result_span)
    buf_ptr.free()

    print("Parent read from child's stdout:", result.strip())
    assert_equal(result, input_string)
    print("I/O Redirection example complete.")


fn main() raises:
    long_running_process_wait()
    print("\n--------------------\n")
    long_running_process_poll()
    print("\n--------------------\n")
    interrupt_process()
    print("\n--------------------\n")
    kill_process()
    print("\n--------------------\n")
    pipe_io_redirect()  # Added call here
