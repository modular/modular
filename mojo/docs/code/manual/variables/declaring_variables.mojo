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


def declaration_intro():
    # start-declaration-intro
    var greeting: String = "Hello World"
    # end-declaration-intro
    _ = greeting


def implicit_variables():
    # start-implicit-variables
    name = "Sam"
    user_id = 0
    # end-implicit-variables
    _, _ = name, user_id


def implicit_variables_annotated():
    # start-implicit-variables-annotated
    name: String = "Sam"
    user_id: Int
    # end-implicit-variables-annotated
    user_id = 0
    _, _ = name, user_id


def explicit_variables():
    # start-explicit-variables
    var name = "Sam"
    var user_id: Int
    # end-explicit-variables
    user_id = 0
    _, _ = name, user_id


def get_name() -> String:
    return "Sam"


def type_annotations():
    # start-type-annotations
    var name: String = get_name()
    # end-type-annotations
    _ = name


def implicit_conversion():
    # start-implicit-conversion
    var temperature: Float64 = 99.0
    print(temperature)
    # end-implicit-conversion
    _ = temperature


# start-late-initialization
def my_function(x: Int):
    var z: Float32
    if x != 0:
        z = 1.0
    else:
        z = foo()
    print(z)


def foo() -> Float32:
    return 3.14
    # end-late-initialization


def main():
    declaration_intro()
    implicit_variables()
    implicit_variables_annotated()
    explicit_variables()
    type_annotations()
    implicit_conversion()
    my_function(1)
