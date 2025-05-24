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

from python import Python
from python import PythonObject


def main():
    var value1: PythonObject = 3.7
    value2 = Python.evaluate("10/3")

    # Compare values
    print("Is value1 greater than 3:", value1 > 3)
    print("Is value1 greater than value2:", value1 > value2)

    # Compare identities
    value3 = value2
    print("value1 is value2:", value1 is value2)
    print("value2 is value3:", value2 is value3)

    # Compare types
    py_float_type = Python.evaluate("float")
    print("Python float type:", py_float_type)
    print("value1 type:", Python.type(value1))
    print("Is value1 a Python float:", Python.type(value1) is py_float_type)
