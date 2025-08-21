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


@fieldwise_init
struct Foobat(Copyable, ExplicitlyCopyable, Movable):
    var x: Int

    fn __copyinit__(out self, other: Self):
        self.x = other.x
        print("__copyinit__")


def main():
    # start-extension-example
    # Without explicit extension: `s` is last used at the print, so it is destroyed after it.
    var s = "abc"
    print(s)  # s.__del__() runs after this line

    # With explicit extension: push last-use to the discard line.
    var t = "xyz"
    print(t)

    # ... some time later
    _ = t  # t.__del__() runs here (after this line)
    # end-extension-example

    var f = Foobat(x=1)
    g = f
    print("before discard")
    _ = f
    print("before second discard")
    _ = g
