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

# start-python-dict-example
from python import Python, PythonObject


def main():
    py_dict = Python.dict()
    py_dict[PythonObject("item_name")] = PythonObject("whizbang")
    py_dict[PythonObject("price")] = PythonObject(11.75)
    py_dict[PythonObject("inventory")] = PythonObject(100)
    print(py_dict)
    # end-python-dict-example
