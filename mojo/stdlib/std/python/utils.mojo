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

# ===----------------------------------------------------------------------=== #
# Standalone implementations of types introduced in:
# https://github.com/modular/modular/pull/5562
#
# Provides NotImplementedError and RichCompareOps for use with Python extension
# modules that require the rich comparison protocol.
# ===----------------------------------------------------------------------=== #


struct RichCompareOps:
    """Flags used by the tp_richcompare function.

    Pass the `op` argument from your rich compare handler to these constants
    to determine which comparison is being requested.

    References:
    - https://github.com/python/cpython/blob/main/Include/object.h#L721
    - https://docs.python.org/3/c-api/typeobj.html#c.PyTypeObject.tp_richcompare
    """

    comptime Py_LT = 0
    comptime Py_LE = 1
    comptime Py_EQ = 2
    comptime Py_NE = 3
    comptime Py_GT = 4
    comptime Py_GE = 5


@fieldwise_init
struct NotImplementedError(TrivialRegisterPassable, Writable):
    """Raise this from a rich compare handler to signal Python's NotImplemented.

    When caught by the `_richcompare_wrapper`, this causes the wrapper to
    return `Py_NotImplemented` to Python rather than setting an exception,
    allowing Python to try the reflected operation on the other operand.

    Example:
        ```mojo
        @staticmethod
        def rich_compare(
            self_ptr: PythonObject, other: PythonObject, op: Int
        ) raises -> Bool:
            if op == RichCompareOps.Py_EQ:
                ...
            raise NotImplementedError()
        ```
    """

    comptime name: String = "NotImplementedError"
    """Well-known name used by `_richcompare_wrapper` for dispatch."""

    def write_to(self, mut writer: Some[Writer]):
        writer.write(Self.name)
