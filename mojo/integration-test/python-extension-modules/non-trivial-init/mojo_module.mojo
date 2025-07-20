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

from os import abort

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with MojoPair type that supports non-trivial initialization.
    """
    try:
        var b = PythonModuleBuilder("mojo_module")

        # Add the MojoPair type with custom initialization and methods
        _ = (
            b.add_type[MojoPair]("MojoPair")
            .def_py_init[MojoPair.pyinit]()
            .def_method[MojoPair.get_first]("get_first")
            .def_method[MojoPair.get_second]("get_second")
            .def_method[MojoPair.swap]("swap")
        )

        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


@fieldwise_init
struct MojoPair(Copyable, Defaultable, Movable, Representable):
    """A pair of integers that can be initialized with custom values."""

    var first: Int
    var second: Int

    fn __init__(out self):
        """Default constructor."""
        self = Self(0, 0)

    fn __init__(out self, args: PythonObject, kwargs: PythonObject) raises:
        """Non-trivial constructor that takes Python arguments."""
        var tuple_len = len(args)
        print(
            "MojoPair.__init__ called with tuple of",
            tuple_len,
            "elements:",
            args,
        )

        if tuple_len != 2:
            raise String(
                "MojoPair requires exactly 2 arguments, got ", tuple_len
            )

        try:
            self.first = Int(args[0])
            self.second = Int(args[1])
        except e:
            raise String("Failed to convert arguments to integers: ", e)

    @staticmethod
    fn pyinit(out self: Self, args: PythonObject, kwargs: PythonObject) raises:
        self = Self(args, kwargs)

    fn __repr__(self) -> String:
        """String representation of the MojoPair."""
        return String("MojoPair(", self.first, ", ", self.second, ")")

    @staticmethod
    fn _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self]:
        """Helper to extract the self pointer from Python object."""
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            return abort[UnsafePointer[Self]](
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type: "
                    ),
                    e,
                )
            )

    @staticmethod
    fn get_first(py_self: PythonObject) -> PythonObject:
        """Get the first value of the pair."""
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].first)

    @staticmethod
    fn get_second(py_self: PythonObject) -> PythonObject:
        """Get the second value of the pair."""
        var self_ptr = Self._get_self_ptr(py_self)
        return PythonObject(self_ptr[].second)

    @staticmethod
    fn swap(py_self: PythonObject) -> PythonObject:
        """Swap the first and second values."""
        var self_ptr = Self._get_self_ptr(py_self)
        var temp = self_ptr[].first
        self_ptr[].first = self_ptr[].second
        self_ptr[].second = temp
        return py_self
