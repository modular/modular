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


def test_mojo_formats_typed_except_bindings():
    source = """\
def typed_handler():
    try:
        fail()
    except Error __as err:
        consume(err)

def inferred_handler():
    try:
        fail()
    except __as err:
        consume(err)
"""
    expected = """\
def typed_handler():
    try:
        fail()
    except Error __as err:
        consume(err)


def inferred_handler():
    try:
        fail()
    except __as err:
        consume(err)
"""
    assert_mojo_format(source, expected)
