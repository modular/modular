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

from python import PythonObject
from os import abort


struct TestStruct(Defaultable, Movable, Representable):
    var a: Int
    var b: Int

    fn __init__(out self):
        self.a = 0
        self.b = 0

    fn __init__(out self, a: Int, b: Int):
        self.a = a
        self.b = b

    fn print(self) -> None:
        print(self.a, self.b)

    fn __repr__(self: TestStruct) -> String:
        return String("TestStruct(", self.a, ", ", self.b, ")")

    @staticmethod
    fn set_a(py_self: PythonObject, a: PythonObject):
        try:
            Self._get_self_ptr(py_self)[].a = Int(a)
        except e:
            abort(String("failed to set a: ", a))

    @staticmethod
    fn set_b(py_self: PythonObject, b: PythonObject):
        try:
            Self._get_self_ptr(py_self)[].b = Int(b)
        except e:
            abort(String("failed to set b: ", b))

    @staticmethod
    fn _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            return abort[UnsafePointer[Self]](
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type:"
                    ),
                    e,
                )
            )
