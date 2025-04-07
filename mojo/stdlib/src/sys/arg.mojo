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
"""Implements functions and variables for interacting with execution and system
environment.

You can import these APIs from the `sys` package. For example:

```mojo
from sys import argv
def main():
    arguments = argv()
    print(
        arguments[0], #app.mojo
        arguments[1]  #Hello world!
    )
    for arg in arguments:
        print(arg)
# If the program is app.mojo:
# mojo run app.mojo "Hello world!"
```
"""


from sys import external_call

from memory import UnsafePointer


# TODO: When we have global variables, this should be a global list.
fn argv() -> VariadicList[StringSlice[StaticConstantOrigin]]:
    """The list of command line arguments.

    Returns:
        The list of command line arguments provided when mojo was invoked.
    """
    # SAFETY:
    #   It is valid to use `StringSlice` here because `StringSlice` is
    #   guaranteed to be ABI compatible with llvm::StringRef.
    var result = VariadicList[StringSlice[StaticConstantOrigin]]("")
    external_call["KGEN_CompilerRT_GetArgV", NoneType](Pointer(to=result))
    return result
