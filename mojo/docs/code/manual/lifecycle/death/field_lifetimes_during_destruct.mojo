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


# start-field-lifetimes-during-destruct
fn consume(var str: String):
    print("Consumed", str)


@fieldwise_init
struct TwoStrings(Copyable, Movable):
    var str1: String
    var str2: String

    fn __del__(deinit self):
        # self value is whole at the beginning of the function
        self.dump()
        # After dump(): str2 is never used again, so str2.__del__() runs now

        consume(self.str1^)
        # self.str1 has been transferred so str1 becomes uninitialized, and
        # no destructor is called for str1.
        # self.__del__() is not called (avoiding an infinite loop).

    fn dump(mut self):
        print("str1:", self.str1)
        print("str2:", self.str2)


def main():
    var two_strings = TwoStrings("foo", "bar")
    # end-field-lifetimes-during-destruct
    _ = two_strings^
