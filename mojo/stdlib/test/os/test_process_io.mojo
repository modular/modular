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
from sys._libc import close
from memory import Span
from testing import assert_equal, assert_true


fn test_stdout_redirect() raises:
    var p = Pipe()
    var process = Process.run("echo", ["hello"], stdout=p.fd_out)
    var status = process.wait()

    assert_true(status.exit_code)
    assert_equal(status.exit_code.value(), 0)

    var buf_size = 16
    var buf_ptr = alloc[UInt8](buf_size)
    var buf_span = Span[mut=True, UInt8](ptr=buf_ptr, length=buf_size)
    var bytes_read = p.read_bytes(buf_span)
    var result_span = buf_span.unsafe_subspan(offset=0, length=Int(bytes_read))
    var result = String(bytes=result_span)
    assert_equal(result, "hello\n")
    buf_ptr.free()


fn test_stderr_redirect() raises:
    var p = Pipe()
    var process = Process.run("sh", ["-c", "echo hello >&2"], stderr=p.fd_out)

    var status = process.wait()

    assert_true(status.exit_code)
    assert_equal(status.exit_code.value(), 0)

    var buf_size = 16
    var buf_ptr = alloc[UInt8](buf_size)
    var buf_span = Span[mut=True, UInt8](ptr=buf_ptr, length=buf_size)
    var bytes_read = p.read_bytes(buf_span)
    var result_span = buf_span.unsafe_subspan(offset=0, length=Int(bytes_read))
    var result = String(bytes=result_span)
    assert_equal(result, "hello\n")
    buf_ptr.free()


fn test_stdin_redirect() raises:
    var p_in = Pipe()
    var p_out = Pipe()

    p_in.write_bytes("hello".as_bytes())
    p_in.set_input_only()  # close the write end for the parent

    var process = Process.run("cat", [], stdin=p_in.fd_in, stdout=p_out.fd_out)

    var status = process.wait()
    assert_true(status.exit_code)
    assert_equal(status.exit_code.value(), 0)

    var buf_size = 16
    var buf_ptr = alloc[UInt8](buf_size)
    var buf_span = Span[mut=True, UInt8](ptr=buf_ptr, length=buf_size)
    var bytes_read = p_out.read_bytes(buf_span)
    var result_span = buf_span.unsafe_subspan(offset=0, length=Int(bytes_read))
    var result = String(bytes=result_span)
    assert_equal(result, "hello")
    buf_ptr.free()


def main():
    test_stdout_redirect()
    test_stderr_redirect()
    test_stdin_redirect()
