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
struct MojoPair(Defaultable, ImplicitlyCopyable, Movable, Representable):
    """A pair of integers that can be initialized with custom values."""

    var first: Int
    var second: Int

    fn __init__(out self):
        """Default constructor."""
        self = Self(0, 0)

    fn __init__(out self, args: PythonObject, kwargs: PythonObject) raises:
        """Non-trivial constructor that takes Python positional and keyword arguments.
        """
        # Check for null arguments before calling len()
        var tuple_len = 0
        var kwargs_len = 0

        if args:
            tuple_len = len(args)
        if kwargs._obj_ptr:
            kwargs_len = len(kwargs)
        print(
            "MojoPair.__init__ called with tuple of",
            tuple_len,
            "elements:",
            args,
            "and kwargs of",
            kwargs_len,
            "elements:",
            kwargs,
        )

        # Handle different argument patterns
        if tuple_len + kwargs_len == 0:
            raise String("MojoPair requires at least 1 argument")

        var first_val: Int
        var second_val: Int

        # Extract positional arguments
        if tuple_len >= 1:
            try:
                first_val = Int(args[0])
            except e:
                raise String("Failed to convert first argument to integer: ", e)
        else:
            first_val = 0  # Default if not provided positionally

        if tuple_len >= 2:
            try:
                second_val = Int(args[1])
            except e:
                raise String(
                    "Failed to convert second argument to integer: ", e
                )
        else:
            second_val = 0  # Default if not provided positionally

        if tuple_len > 2:
            raise String(
                "MojoPair accepts at most 2 positional arguments, got ",
                tuple_len,
            )

        # Process keyword arguments - they override positional arguments
        if kwargs_len > 0:
            try:
                # Check for 'first' keyword argument
                if "first" in kwargs:
                    first_val = Int(kwargs["first"])

                # Check for 'second' keyword argument
                if "second" in kwargs:
                    second_val = Int(kwargs["second"])
            except e:
                raise String("Failed to process keyword arguments: ", e)

        # Ensure we have valid values for both
        if tuple_len == 0 and kwargs_len > 0:
            # Pure keyword argument case - need both
            if "first" not in kwargs or "second" not in kwargs:
                raise String(
                    "When using only keyword arguments, both 'first' and"
                    " 'second' must be provided"
                )
        elif tuple_len == 1 and kwargs_len > 0:
            # Mixed case with one positional - need 'second' in kwargs
            if "second" not in kwargs:
                raise String(
                    "When providing 1 positional argument, 'second' must be"
                    " provided as keyword argument"
                )
        elif tuple_len == 1 and kwargs_len == 0:
            # Single positional argument case - need exactly 2
            raise String(
                "MojoPair requires exactly 2 arguments when using only"
                " positional arguments"
            )

        self.first = first_val
        self.second = second_val

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
