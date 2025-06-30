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

from common import TestStruct
from python import PythonObject
from python.bindings import PythonModuleBuilder
from os import abort


@export
fn PyInit_mojo_module_b() -> PythonObject:
    try:
        var m = PythonModuleBuilder("mojo_module_b")
        m.def_function[print_test_struct]("print_test_struct")
        m.def_function[add]("add")
        return m.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


fn print_test_struct(s: PythonObject) -> None:
    var self_ptr = TestStruct._get_self_ptr(s)
    self_ptr[].print()


fn add(s: PythonObject) -> PythonObject:
    var self_ptr = TestStruct._get_self_ptr(s)
    return self_ptr[].a + self_ptr[].b
