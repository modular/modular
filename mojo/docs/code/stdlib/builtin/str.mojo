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

from testing import assert_true


@fieldwise_init
struct Foo(Stringable):
    var s: String

    fn __str__(self) -> String:
        return self.s


def main():
    var foo = Foo("test")
    print(String(foo) == "test")  # True

    # --------
    # Not in docstring:
    assert_true(String(foo) == "test")
    # --------
