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

# start-python-to-mojo-conversions
from python import Python
from python import PythonObject


def main():
    var py_string = PythonObject("Hello, Mojo!")
    var py_bool = PythonObject(True)
    var py_int = PythonObject(123)
    var py_float = PythonObject(3.14)

    var mojo_string = String(py_string)
    var mojo_bool = Bool(py_bool)
    var mojo_int = Int(py_int)
    var mojo_float = Float64(py_float)
    # end-python-to-mojo-conversions
    _ = mojo_string^
    _ = mojo_bool
    _ = mojo_int
    _ = mojo_float
