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


# start-lexical-scopes
def lexical_scopes():
    var num = 1
    var dig = 1
    if num == 1:
        print("num:", num)  # Reads the outer-scope "num"
        var num = 2  # Creates new inner-scope "num"
        print("num:", num)  # Reads the inner-scope "num"
        dig = 2  # Updates the outer-scope "dig"
    print("num:", num)  # Reads the outer-scope "num"
    print("dig:", dig)  # Reads the outer-scope "dig"
    # end-lexical-scopes


# start-function-scopes
def function_scopes():
    num = 1
    if num == 1:
        print(num)  # Reads the function-scope "num"
        num = 2  # Updates the function-scope variable
        print(num)  # Reads the function-scope "num"
    print(num)  # Reads the function-scope "num"
    # end-function-scopes


def main():
    lexical_scopes()
    function_scopes()
