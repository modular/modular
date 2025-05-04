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
# RUN: %mojo %s


from json import loads
from testing import assert_equal


def test_loads():
    var o1 = loads(
        """
    {
        "name": "John",
        "age": 30,
        "city": "New York"
    }"""
    )
    assert_equal(
        String(o1), '{"name": "John", "age": 30.0, "city": "New York"}'
    )

    assert_equal(String(loads("[1.0, 2.0, 3.0]")), "[1.0, 2.0, 3.0]")

    assert_equal(String(loads("true")), "true")

    assert_equal(String(loads("false")), "false")

    assert_equal(String(loads("null")), "null")

    assert_equal(String(loads("123")), "123.0")

    assert_equal(String(loads("123.456")), "123.456")

    assert_equal(String(loads("123.456e3")), "123456.0")


def main():
    test_loads()
