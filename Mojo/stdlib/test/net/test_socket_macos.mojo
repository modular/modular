# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.ffi import c_int
from std.testing import TestSuite, assert_not_equal

from std.net._socket import Socket
from std.net._sys.macos import SO_NOSIGPIPE
from std.net._sys.unix import AF_INET, SOCK_STREAM, SOL_SOCKET


def test_no_sigpipe() raises:
    var sock = Socket(domain=AF_INET, type=SOCK_STREAM)
    var no_sigpipe: c_int
    try:
        no_sigpipe = sock.option(SOL_SOCKET, SO_NOSIGPIPE)
    finally:
        sock^.close()
    assert_not_equal(no_sigpipe, 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
