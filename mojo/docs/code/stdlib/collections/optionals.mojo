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


fn empty_optional():
    instance = Optional[String]()
    print(instance)  # Output: None


fn emptier_optional():
    instance = Optional[String](None)
    print(instance)  # Output: None


fn optional():
    instance = Optional[String]("Hello")
    print(instance)  # Output: 'Hello'


fn bounds():
    empty_instance = Optional[Int]()
    populated_instance = Optional[Int](50)

    # Bounds returns a tuple: (`bounds`, `Optional` version of `bounds`)
    # with the length of the `Optional`.
    print(empty_instance.bounds()[0])  # 0
    print(populated_instance.bounds()[0])  # 1
    print(empty_instance.bounds()[1])  # 0
    print(populated_instance.bounds()[1])  # 1


fn value():
    instance = Optional("Hello")
    x = instance.value()
    print(x)  # Hello
    # instance = Optional[String]() # Uncomment both lines to crash
    # print(instance.value())


fn unsafe_value():
    instance = Optional("Hello")
    x = instance.unsafe_value()
    print(x)  # Hello
    instance = Optional[String](None)

    # Best practice:
    if instance:
        y = instance.unsafe_value()  # Will not reach this line
        print(y)

    # May crash, may access re-used uninitialized memory
    # y = instance.unsafe_value() # Danger: No safety guards for `None`
    # print(y)


fn take():
    instance = Optional("Hello")
    print(instance.bounds()[0])  # Output: 1
    x = instance.take()  # Moves value from `instance` to `x`
    print(x)  # Output: Hello

    # `instance` is now `Optional(None)`
    print(instance.bounds()[0])  # Output: 0
    print(instance)  # Output: None

    # Best practice
    if instance:
        y = instance.take()  # Won't reach this line
        print(y)

    # Used directly
    # y = instance.take()         # ABORT: `Optional.take()` called on empty `Optional` (via runtime `abort`)
    # print(y)                    # Does not reach this line


fn unsafe_take():
    instance = Optional("Hello")
    print(instance.bounds()[0])  # Output: 1
    x = instance.unsafe_take()  # Moves value from `instance` to `x`
    print(x)  # Output: Hello

    # `instance` is now `Optional(None)`
    print(instance.bounds()[0])  # Output: 0
    print(instance)  # Output: None

    # Best practice:
    if instance:
        y = instance.unsafe_take()  # Won't reach this line
        print(y)

    # May crash:
    # y = (
    #     instance.unsafe_take()
    # )  # ABORT: `Optional.take()` called on empty `Optional` (via `debug_assert`)
    # print(y)  # Does not reach this line


fn or_else():
    instance = Optional("Hello")
    print(instance)  # Output: 'Hello'
    print(instance.or_else("Bye"))  # Output: Hello
    instance = None
    print(instance)  # Output: None
    print(instance.or_else("Bye"))  # Output: Bye


fn iter():
    instance = Optional("Hello")
    for value in instance:
        print(value)  # Output: Hello
    instance = None
    for value in instance:
        print(value)  # Does not reach line


fn main():
    empty_optional()
    optional()
    emptier_optional()
    bounds()
    value()
    unsafe_value()
    take()
    unsafe_take()
    or_else()
    iter()
