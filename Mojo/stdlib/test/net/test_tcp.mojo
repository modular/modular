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

from std.net.tcp import TCPSocket
from std.net.tcp.option import KeepAlive, NoDelay, ReuseAddress, ReusePort
from std.testing import TestSuite, assert_false, assert_true


def test_socket_options_round_trip() raises:
    var socket = TCPSocket.v4()

    assert_false(socket.get_option[ReuseAddress]())
    socket.set_option[ReuseAddress](True)
    assert_true(socket.get_option[ReuseAddress]())

    assert_false(socket.get_option[ReusePort]())
    socket.set_option[ReusePort](True)
    assert_true(socket.get_option[ReusePort]())

    assert_false(socket.get_option[NoDelay]())
    socket.set_option[NoDelay](True)
    assert_true(socket.get_option[NoDelay]())

    assert_false(socket.get_option[KeepAlive]())
    socket.set_option[KeepAlive](True)
    assert_true(socket.get_option[KeepAlive]())


def test_ipv6_socket_options() raises:
    var socket = TCPSocket.v6()
    assert_false(socket.get_option[NoDelay]())
    socket.set_option[NoDelay](True)
    assert_true(socket.get_option[NoDelay]())


def test_explicit_close() raises:
    TCPSocket.v4().close()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
