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


def main():
    message = "Hello"  # type = String
    alias name = " Pat"  # type = StringLiteral
    greeting = " good Day!"  # type = String

    # Mutate the original `message` String
    message += name
    message += greeting
    print(message)

    alias last_name = "Curie"

    # Compile-time StringLiteral alias
    alias marie = "Marie " + last_name
    print(marie)

    # Compile-time concatenation before materializing to a run-time `String`
    pierre = "Pierre " + last_name
    print(pierre)

    x, y = 3, 4
    result = "The point at (" + String(x) + ", " + String(y) + ")"
    result2 = String("The point at (", x, ", ", y, ")")
    print(result == result2)

    # String replication
    var str1: String = "la"
    str2 = str1 * 5
    print(str2)

    alias divider1 = "=" * 40
    alias symbol = "#"
    alias divider2 = symbol * 40

    # You must define the following function using `fn` because an alias
    # initializer cannot call a function that can potentially raise an error.
    fn generate_divider(char: String, repeat: Int) -> String:
        return char * repeat

    alias divider3 = generate_divider("~", 40)  # Evaluated at compile-time

    print(divider1)
    print(divider2)
    print(divider3)

    repeat = 40
    div1 = "^" * repeat
    print(div1)
    print("_" * repeat)

    # String comparison
    var animal: String = "bird"

    is_cat_eq = "cat" == animal
    print('Is "cat" equal to "{}"?'.format(animal), is_cat_eq)

    is_cat_ne = "cat" != animal
    print('Is "cat" not equal to "{}"?'.format(animal), is_cat_ne)

    is_bird_eq = "bird" == animal
    print('Is "bird" equal to "{}"?'.format(animal), is_bird_eq)

    is_cat_gt = "CAT" > animal
    print('Is "CAT" greater than "{}"?'.format(animal), is_cat_gt)

    is_ge_cat = animal >= "CAT"
    print('Is "{}" greater than or equal to "CAT"?'.format(animal), is_ge_cat)

    # Substring testing
    var food: String = "peanut butter"

    if "nut" in food:
        print("It contains a nut")
    else:
        print("It doesn't contain a nut")

    # String indexing and slicing
    var alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # String type value
    print(alphabet[0], alphabet[-1])

    # The following would produce a run-time error
    # print(alphabet[45])

    print(alphabet[1:4])  # The 2nd through 4th characters
    print(alphabet[:6])  # The first 6 characters
    print(alphabet[-6:])  # The last 6 characters

    # TODO: the current example in "Assignment expressions" isn't
    # readily testable, because it uses command-line input.
