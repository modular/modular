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

from sys.intrinsics import _type_is_eq

from python import PythonObject as PO  # for brevity of signatures below
from python.bindings import check_arguments_arity


struct PyObjectFunction[func_type: AnyTrivialRegType, is_method: Bool]:
    """Wrapper to hide the binding logic for functions taking a variadic number
    of PythonObject arguments.

    This currently supports function types with up to 3 positional arguments,
    as well as raising functions, and both functions that return a PythonObject
    or nothing.

    Note:
        This is a private implementation detail of the Python bindings, and have
        been designed to make it easier to add support for higher argument
        arities in the future.
    """

    var _func: func_type

    # ===-------------------------------------------------------------------===#
    # 0 arguments
    # ===-------------------------------------------------------------------===#

    alias _0er = fn () raises -> PO
    alias _0r = fn () -> PO
    alias _0e = fn () raises
    alias _0 = fn ()

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0er, False], f: Self._0er):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0r, False], f: Self._0r):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0e, False], f: Self._0e):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0, False], f: Self._0):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 1 argument
    # ===-------------------------------------------------------------------===#

    alias _1er = fn (PO) raises -> PO
    alias _1r = fn (PO) -> PO
    alias _1e = fn (PO) raises
    alias _1 = fn (PO)

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._1er, is_method], f: Self._1er):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._1r, is_method], f: Self._1r):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._1e, is_method], f: Self._1e):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._1, is_method], f: Self._1):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 2 arguments
    # ===-------------------------------------------------------------------===#

    alias _2er = fn (PO, PO) raises -> PO
    alias _2r = fn (PO, PO) -> PO
    alias _2e = fn (PO, PO) raises
    alias _2 = fn (PO, PO)

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._2er, is_method], f: Self._2er):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._2r, is_method], f: Self._2r):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._2e, is_method], f: Self._2e):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._2, is_method], f: Self._2):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 3 arguments
    # ===-------------------------------------------------------------------===#

    alias _3er = fn (PO, PO, PO) raises -> PO
    alias _3r = fn (PO, PO, PO) -> PO
    alias _3e = fn (PO, PO, PO) raises
    alias _3 = fn (PO, PO, PO)

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._3er, is_method], f: Self._3er):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._3r, is_method], f: Self._3r):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._3e, is_method], f: Self._3e):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._3, is_method], f: Self._3):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # Compile-time check utilities
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    fn _has_type[other_func_type: AnyTrivialRegType]() -> Bool:
        return _type_is_eq[func_type, other_func_type]()

    @staticmethod
    @always_inline("nodebug")
    fn _has_arity(arity: Int) -> Bool:
        @parameter
        if (
            Self._has_type[Self._0er]()
            or Self._has_type[Self._0r]()
            or Self._has_type[Self._0e]()
            or Self._has_type[Self._0]()
        ):
            return arity == 0
        elif (
            Self._has_type[Self._1er]()
            or Self._has_type[Self._1r]()
            or Self._has_type[Self._1e]()
            or Self._has_type[Self._1]()
        ):
            return arity == 1
        elif (
            Self._has_type[Self._2er]()
            or Self._has_type[Self._2r]()
            or Self._has_type[Self._2e]()
            or Self._has_type[Self._2]()
        ):
            return arity == 2
        elif (
            Self._has_type[Self._3er]()
            or Self._has_type[Self._3r]()
            or Self._has_type[Self._3e]()
            or Self._has_type[Self._3]()
        ):
            return arity == 3
        else:
            return False

    # ===-------------------------------------------------------------------===#
    # Call wrappers
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn _call_func(self, py_args: PO) raises -> PO:
        @parameter
        if Self._has_arity(0):
            check_arguments_arity(0, py_args)

            @parameter
            if self._has_type[Self._0er]():
                return rebind[Self._0er](self._func)()
            elif self._has_type[Self._0r]():
                return rebind[Self._0r](self._func)()
            elif self._has_type[Self._0e]():
                return rebind[Self._0e](self._func)()
            elif self._has_type[Self._0]():
                return rebind[Self._0](self._func)()
        elif Self._has_arity(1):
            check_arguments_arity(1, py_args)
            var arg0 = py_args[0]

            @parameter
            if self._has_type[Self._1er]():
                return rebind[Self._1er](self._func)(arg0)
            elif self._has_type[Self._1r]():
                return rebind[Self._1r](self._func)(arg0)
            elif self._has_type[Self._1e]():
                return rebind[Self._1e](self._func)(arg0)
            elif self._has_type[Self._1]():
                return rebind[Self._1](self._func)(arg0)
        elif Self._has_arity(2):
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]

            @parameter
            if self._has_type[Self._2er]():
                return rebind[Self._2er](self._func)(arg0, arg1)
            elif self._has_type[Self._2r]():
                return rebind[Self._2r](self._func)(arg0, arg1)
            elif self._has_type[Self._2e]():
                return rebind[Self._2e](self._func)(arg0, arg1)
            elif self._has_type[Self._2]():
                return rebind[Self._2](self._func)(arg0, arg1)
        elif Self._has_arity(3):
            check_arguments_arity(3, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]

            @parameter
            if self._has_type[Self._3er]():
                return rebind[Self._3er](self._func)(arg0, arg1, arg2)
            elif self._has_type[Self._3r]():
                return rebind[Self._3r](self._func)(arg0, arg1, arg2)
            elif self._has_type[Self._3e]():
                return rebind[Self._3e](self._func)(arg0, arg1, arg2)
            elif self._has_type[Self._3]():
                return rebind[Self._3](self._func)(arg0, arg1, arg2)

        constrained[False, "unsupported arity or signature"]()
        return PO()

    @always_inline("nodebug")
    fn _call_method(self, py_self: PO, py_args: PO) raises -> PO:
        constrained[not Self._has_arity(0), "method arity must not be 0"]()

        @parameter
        if Self._has_arity(1):
            check_arguments_arity(0, py_args)

            @parameter
            if self._has_type[Self._1er]():
                return rebind[Self._1er](self._func)(py_self)
            elif self._has_type[Self._1r]():
                return rebind[Self._1r](self._func)(py_self)
            elif self._has_type[Self._1e]():
                return rebind[Self._1e](self._func)(py_self)
            elif self._has_type[Self._1]():
                return rebind[Self._1](self._func)(py_self)
        elif Self._has_arity(2):
            check_arguments_arity(1, py_args)
            var arg0 = py_args[0]

            @parameter
            if self._has_type[Self._2er]():
                return rebind[Self._2er](self._func)(py_self, arg0)
            elif self._has_type[Self._2r]():
                return rebind[Self._2r](self._func)(py_self, arg0)
            elif self._has_type[Self._2e]():
                return rebind[Self._2e](self._func)(py_self, arg0)
            elif self._has_type[Self._2]():
                return rebind[Self._2](self._func)(py_self, arg0)
        elif Self._has_arity(3):
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]

            @parameter
            if self._has_type[Self._3er]():
                return rebind[Self._3er](self._func)(py_self, arg0, arg1)
            elif self._has_type[Self._3r]():
                return rebind[Self._3r](self._func)(py_self, arg0, arg1)
            elif self._has_type[Self._3e]():
                return rebind[Self._3e](self._func)(py_self, arg0, arg1)
            elif self._has_type[Self._3]():
                return rebind[Self._3](self._func)(py_self, arg0, arg1)

        constrained[False, "unsupported arity or signature"]()
        return PO()
