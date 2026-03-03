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

from std.testing import assert_true


@fieldwise_init
struct Foo(Writable):
    var s: String

    fn write_to(self, mut writer: Some[Writer]):
        writer.write_string(self.s)


def main() raises:
    var foo = Foo("test")
    print(String(foo) == "test")  # True

    # --------
    # Not in docstring:
    assert_true(String(foo) == "test")
    # --------
