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

from tests.util import assert_mojo_format


def test_decorated_var():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var  x:  Int\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
    )
    assert_mojo_format(source, expected)


def test_decorated_var_multiple_fields():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "    @doc_hidden\n"
        "    var y: String\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "\n"
        "    @doc_hidden\n"
        "    var y: String\n"
    )
    assert_mojo_format(source, expected)


def test_decorated_var_trailing_comment():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int  # comment\n"
        "\n"
        "\n"
        "def f():\n"
        "    @__address_space(.SHARED)\n"
        "    var y = 0  # comment\n"
    )
    assert_mojo_format(source, source)


def test_decorated_local_var_spacing():
    source = (
        "def f():\n"
        "    var a = 1\n"
        "    @__address_space(.SHARED)\n"
        "    var x = 0\n"
        "    if a:\n"
        "        @__address_space(.SHARED)\n"
        "        var y = 0\n"
        "    var b = 2\n"
        "\n"
        "    @__address_space(.SHARED)\n"
        "    var z = 0\n"
    )
    assert_mojo_format(source, source)


def test_decorated_var_in_nested_struct_keeps_spacing():
    source = (
        "def f():\n"
        "    struct Foo:\n"
        "        var a: Int\n"
        "        @doc_hidden\n"
        "        var x: Int\n"
    )
    expected = (
        "def f():\n"
        "    struct Foo:\n"
        "        var a: Int\n"
        "\n"
        "        @doc_hidden\n"
        "        var x: Int\n"
    )
    assert_mojo_format(source, expected)


def test_stacked_decorators_on_local_var():
    source = (
        "def f():\n"
        "    @__address_space(.SHARED)\n"
        "\n"
        "    @foo\n"
        "    var x = 0\n"
    )
    expected = (
        "def f():\n"
        "    @__address_space(.SHARED)\n"
        "    @foo\n"
        "    var x = 0\n"
    )
    assert_mojo_format(source, expected)


def test_blank_line_between_decorator_and_var():
    source = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "\n"
        "    var x: Int\n"
        "\n"
        "\n"
        "def f():\n"
        "    @__address_space(.SHARED)\n"
        "\n"
        "    var y = 0\n"
    )
    expected = (
        "struct Foo:\n"
        "    @doc_hidden\n"
        "    var x: Int\n"
        "\n"
        "\n"
        "def f():\n"
        "    @__address_space(.SHARED)\n"
        "    var y = 0\n"
    )
    assert_mojo_format(source, expected)
