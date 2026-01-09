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
struct Person(Stringable):
    var name: String
    var age: Int

    fn __str__(self) -> String:
        return self.name + " (" + String(self.age) + ")"


def string_summary():
    from testing import assert_equal, assert_raises

    var person = Person("Alice", 30)
    print(String(person))  # => Alice (30)

    # --------
    # Not in docstring:
    assert_equal(String(person), "Alice (30)")
    # --------

    var text = "Hello"

    # String properties and indexing
    print(len(text))  # 5
    print(text[1])  # e
    print(text[-1])  # o

    # In-place concatenation
    text += " World"
    print(text)

    # Searching and checking
    if "World" in text:
        print("Found 'World' in text")

    var pos = text.find("World")
    if pos != -1:
        print("'World' found at position:", pos)

    # String replacement
    var replaced = text.replace("Hello", "Hi")  # "Hi World"
    print(replaced)

    # String formatting
    var name = "Alice"
    var age = 30
    var formatted = "{} is {} years old".format(name, age)
    print(formatted)  # "Alice is 30 years old"

    # --------
    # Not in docstring:
    assert_equal(replaced, "Hi World")
    assert_equal(formatted, "Alice is 30 years old")
    # --------


def main():
    string_summary()
