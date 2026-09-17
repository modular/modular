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

# RUN: %mojo %s


def test_parenthesized_tstring[x: Int]():
    # fmt: off
    comptime assert x > 0, (t"expected positive number, got {x}")
    # fmt: on


def test_nested_parenthesized_tstring[x: Int]():
    # fmt: off
    comptime assert x > 0, ((t"expected positive number, got {x}"))
    # fmt: on


def main():
    test_parenthesized_tstring[1]()
    test_nested_parenthesized_tstring[1]()
