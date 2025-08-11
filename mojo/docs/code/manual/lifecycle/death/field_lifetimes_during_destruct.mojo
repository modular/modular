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


fn consume(var str: String):
    print("Consumed", str)


@fieldwise_init
struct TwoStrings:
    var str1: String
    var str2: String

    fn __moveinit__(out self, deinit existing: Self):
        self.str1 = existing.str1^
        self.str2 = existing.str2^

    fn __del__(deinit self):
        self.dump()  # Self is still whole here
        # Mojo calls self.str2.__del__() since str2 isn't used anymore

        consume(self.str1^)
        # self.str1 has been transferred so it is also destroyed now;
        # `self.__del__()` is not called (avoiding an infinite loop).

    fn dump(mut self):
        print("str1:", self.str1)
        print("str2:", self.str2)


def main():
    var two_strings = TwoStrings("foo", "bar")
