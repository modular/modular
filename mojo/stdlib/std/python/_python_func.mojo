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

from std.collections import OwnedKwargsDict
from std.os import abort
from std.sys.intrinsics import _type_is_eq

from std.python import PythonObject as PO  # for brevity of signatures below
from std.python.bindings import check_arguments_arity


struct PyObjectFunction[
    func_type: TrivialRegisterPassable,
    self_type: ImplicitlyDestructible = NoneType,
    has_kwargs: Bool = False,
](ImplicitlyCopyable):
    """Wrapper to hide the binding logic for functions taking a variadic number
    of PythonObject arguments.

    This currently supports function types with up to 8 positional arguments,
    both functions that raise and those that don't, both functions that return a PythonObject
    or nothing, and both functions that accept keyword arguments and those that don't.

    The self_type parameter controls self parameter handling:
    - NoneType (default): No self parameter expected
    - PythonObject: Self parameter is a PythonObject (for methods)
    - Other types: Self parameter will be auto-downcast from PythonObject

    The has_kwargs parameter indicates whether this function accepts keyword arguments:
    - False (default): Function does not accept kwargs
    - True: Function's last parameter is OwnedKwargsDict[PythonObject]

    Note:
        This is a private implementation detail of the Python bindings, and have
        been designed to make it easier to add support for higher argument
        arities in the future.
    """

    var _func: Self.func_type

    # ===-------------------------------------------------------------------===#
    # 0 arguments
    # ===-------------------------------------------------------------------===#

    comptime _0er = def() thin raises -> PO
    comptime _0r = def() thin -> PO
    comptime _0e = def() thin raises
    comptime _0 = def() thin

    comptime _0er_kwargs = def(OwnedKwargsDict[PO]) thin raises -> PO
    comptime _0r_kwargs = def(OwnedKwargsDict[PO]) thin -> PO
    comptime _0e_kwargs = def(OwnedKwargsDict[PO]) thin raises
    comptime _0_kwargs = def(OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(out self: PyObjectFunction[Self._0er], f: Self._0er):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(out self: PyObjectFunction[Self._0r], f: Self._0r):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(out self: PyObjectFunction[Self._0e], f: Self._0e):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(out self: PyObjectFunction[Self._0], f: Self._0):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._0er_kwargs, has_kwargs=True],
        f: Self._0er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._0r_kwargs, has_kwargs=True],
        f: Self._0r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._0e_kwargs, has_kwargs=True],
        f: Self._0e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._0_kwargs, has_kwargs=True],
        f: Self._0_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 1 argument
    # ===-------------------------------------------------------------------===#

    comptime _1er = def(PO) thin raises -> PO
    comptime _1r = def(PO) thin -> PO
    comptime _1e = def(PO) thin raises
    comptime _1 = def(PO) thin

    comptime _1er_kwargs = def(PO, OwnedKwargsDict[PO]) thin raises -> PO
    comptime _1r_kwargs = def(PO, OwnedKwargsDict[PO]) thin -> PO
    comptime _1e_kwargs = def(PO, OwnedKwargsDict[PO]) thin raises
    comptime _1_kwargs = def(PO, OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1er, Self.self_type], f: Self._1er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1r, Self.self_type], f: Self._1r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1e, Self.self_type], f: Self._1e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1, Self.self_type], f: Self._1
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 1 argument (typed self methods - 0 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _1er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin]
    ) thin raises -> PO
    comptime _1r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin]
    ) thin -> PO
    comptime _1e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin]
    ) thin raises
    comptime _1_self = def(UnsafePointer[Self.self_type, MutAnyOrigin]) thin

    comptime _1er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _1r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _1e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) thin raises
    comptime _1_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1er_self, Self.self_type],
        f: Self._1er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1r_self, Self.self_type],
        f: Self._1r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1e_self, Self.self_type],
        f: Self._1e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._1_self, Self.self_type],
        f: Self._1_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._1_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 2 arguments (typed self methods - 1 additional argument)
    # ===-------------------------------------------------------------------===#

    comptime _2er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO
    ) thin raises -> PO
    comptime _2r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO
    ) thin -> PO
    comptime _2e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO
    ) thin raises
    comptime _2_self = def(UnsafePointer[Self.self_type, MutAnyOrigin], PO) thin

    comptime _2er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _2r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _2e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) thin raises
    comptime _2_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2er_self, Self.self_type],
        f: Self._2er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2r_self, Self.self_type],
        f: Self._2r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2e_self, Self.self_type],
        f: Self._2e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2_self, Self.self_type],
        f: Self._2_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 3 arguments (typed self methods - 2 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _3er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) thin raises -> PO
    comptime _3r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) thin -> PO
    comptime _3e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) thin raises
    comptime _3_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) thin

    comptime _3er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _3r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _3e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) thin raises
    comptime _3_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3er_self, Self.self_type],
        f: Self._3er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3r_self, Self.self_type],
        f: Self._3r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3e_self, Self.self_type],
        f: Self._3e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3_self, Self.self_type],
        f: Self._3_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 4 arguments (typed self methods - 3 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _4er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) thin raises -> PO
    comptime _4r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) thin -> PO
    comptime _4e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) thin raises
    comptime _4_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) thin

    comptime _4er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises -> PO
    comptime _4r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin -> PO
    comptime _4e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises
    comptime _4_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4er_self, Self.self_type],
        f: Self._4er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4r_self, Self.self_type],
        f: Self._4r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4e_self, Self.self_type],
        f: Self._4e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4_self, Self.self_type],
        f: Self._4_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 2 arguments
    # ===-------------------------------------------------------------------===#

    comptime _2er = def(PO, PO) thin raises -> PO
    comptime _2r = def(PO, PO) thin -> PO
    comptime _2e = def(PO, PO) thin raises
    comptime _2 = def(PO, PO) thin

    comptime _2er_kwargs = def(PO, PO, OwnedKwargsDict[PO]) thin raises -> PO
    comptime _2r_kwargs = def(PO, PO, OwnedKwargsDict[PO]) thin -> PO
    comptime _2e_kwargs = def(PO, PO, OwnedKwargsDict[PO]) thin raises
    comptime _2_kwargs = def(PO, PO, OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2er, Self.self_type], f: Self._2er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2r, Self.self_type], f: Self._2r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2e, Self.self_type], f: Self._2e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._2, Self.self_type], f: Self._2
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._2_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 3 arguments
    # ===-------------------------------------------------------------------===#

    comptime _3er = def(PO, PO, PO) thin raises -> PO
    comptime _3r = def(PO, PO, PO) thin -> PO
    comptime _3e = def(PO, PO, PO) thin raises
    comptime _3 = def(PO, PO, PO) thin

    comptime _3er_kwargs = def(
        PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _3r_kwargs = def(PO, PO, PO, OwnedKwargsDict[PO]) thin -> PO
    comptime _3e_kwargs = def(PO, PO, PO, OwnedKwargsDict[PO]) thin raises
    comptime _3_kwargs = def(PO, PO, PO, OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3er, Self.self_type], f: Self._3er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3r, Self.self_type], f: Self._3r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3e, Self.self_type], f: Self._3e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._3, Self.self_type], f: Self._3
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._3_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 4 arguments
    # ===-------------------------------------------------------------------===#

    comptime _4er = def(PO, PO, PO, PO) thin raises -> PO
    comptime _4r = def(PO, PO, PO, PO) thin -> PO
    comptime _4e = def(PO, PO, PO, PO) thin raises
    comptime _4 = def(PO, PO, PO, PO) thin

    comptime _4er_kwargs = def(
        PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _4r_kwargs = def(PO, PO, PO, PO, OwnedKwargsDict[PO]) thin -> PO
    comptime _4e_kwargs = def(PO, PO, PO, PO, OwnedKwargsDict[PO]) thin raises
    comptime _4_kwargs = def(PO, PO, PO, PO, OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4er, Self.self_type], f: Self._4er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4r, Self.self_type], f: Self._4r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4e, Self.self_type], f: Self._4e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._4, Self.self_type], f: Self._4
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._4_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 5 arguments (typed self methods - 4 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _5er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) thin raises -> PO
    comptime _5r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) thin -> PO
    comptime _5e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) thin raises
    comptime _5_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) thin

    comptime _5er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises -> PO
    comptime _5r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin -> PO
    comptime _5e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises
    comptime _5_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5er_self, Self.self_type],
        f: Self._5er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5r_self, Self.self_type],
        f: Self._5r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5e_self, Self.self_type],
        f: Self._5e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5_self, Self.self_type],
        f: Self._5_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 5 arguments
    # ===-------------------------------------------------------------------===#

    comptime _5er = def(PO, PO, PO, PO, PO) thin raises -> PO
    comptime _5r = def(PO, PO, PO, PO, PO) thin -> PO
    comptime _5e = def(PO, PO, PO, PO, PO) thin raises
    comptime _5 = def(PO, PO, PO, PO, PO) thin

    comptime _5er_kwargs = def(
        PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _5r_kwargs = def(
        PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _5e_kwargs = def(
        PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises
    comptime _5_kwargs = def(PO, PO, PO, PO, PO, OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5er, Self.self_type], f: Self._5er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5r, Self.self_type], f: Self._5r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5e, Self.self_type], f: Self._5e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._5, Self.self_type], f: Self._5
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._5_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 6 arguments (typed self methods - 5 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _6er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) thin raises -> PO
    comptime _6r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) thin -> PO
    comptime _6e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) thin raises
    comptime _6_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) thin

    comptime _6er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises -> PO
    comptime _6r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin -> PO
    comptime _6e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises
    comptime _6_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6er_self, Self.self_type],
        f: Self._6er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6r_self, Self.self_type],
        f: Self._6r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6e_self, Self.self_type],
        f: Self._6e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6_self, Self.self_type],
        f: Self._6_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 6 arguments
    # ===-------------------------------------------------------------------===#

    comptime _6er = def(PO, PO, PO, PO, PO, PO) thin raises -> PO
    comptime _6r = def(PO, PO, PO, PO, PO, PO) thin -> PO
    comptime _6e = def(PO, PO, PO, PO, PO, PO) thin raises
    comptime _6 = def(PO, PO, PO, PO, PO, PO) thin

    comptime _6er_kwargs = def(
        PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _6r_kwargs = def(
        PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _6e_kwargs = def(
        PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises
    comptime _6_kwargs = def(PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6er, Self.self_type], f: Self._6er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6r, Self.self_type], f: Self._6r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6e, Self.self_type], f: Self._6e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._6, Self.self_type], f: Self._6
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._6_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 7 arguments (typed self methods - 6 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _7er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO
    ) thin raises -> PO
    comptime _7r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO
    ) thin -> PO
    comptime _7e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO
    ) thin raises
    comptime _7_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO
    ) thin

    comptime _7er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises -> PO
    comptime _7r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin -> PO
    comptime _7e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises
    comptime _7_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7er_self, Self.self_type],
        f: Self._7er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7r_self, Self.self_type],
        f: Self._7r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7e_self, Self.self_type],
        f: Self._7e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7_self, Self.self_type],
        f: Self._7_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 7 arguments
    # ===-------------------------------------------------------------------===#

    comptime _7er = def(PO, PO, PO, PO, PO, PO, PO) thin raises -> PO
    comptime _7r = def(PO, PO, PO, PO, PO, PO, PO) thin -> PO
    comptime _7e = def(PO, PO, PO, PO, PO, PO, PO) thin raises
    comptime _7 = def(PO, PO, PO, PO, PO, PO, PO) thin

    comptime _7er_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _7r_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _7e_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises
    comptime _7_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7er, Self.self_type], f: Self._7er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7r, Self.self_type], f: Self._7r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7e, Self.self_type], f: Self._7e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._7, Self.self_type], f: Self._7
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._7_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._7_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 8 arguments (typed self methods - 7 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _8er_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO, PO
    ) thin raises -> PO
    comptime _8r_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO, PO
    ) thin -> PO
    comptime _8e_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO, PO
    ) thin raises
    comptime _8_self = def(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO, PO, PO
    ) thin

    comptime _8er_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises -> PO
    comptime _8r_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin -> PO
    comptime _8e_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin raises
    comptime _8_self_kwargs = def(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8er_self, Self.self_type],
        f: Self._8er_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8r_self, Self.self_type],
        f: Self._8r_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8e_self, Self.self_type],
        f: Self._8e_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8_self, Self.self_type],
        f: Self._8_self,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8er_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8r_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8e_self_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 8 arguments
    # ===-------------------------------------------------------------------===#

    comptime _8er = def(PO, PO, PO, PO, PO, PO, PO, PO) thin raises -> PO
    comptime _8r = def(PO, PO, PO, PO, PO, PO, PO, PO) thin -> PO
    comptime _8e = def(PO, PO, PO, PO, PO, PO, PO, PO) thin raises
    comptime _8 = def(PO, PO, PO, PO, PO, PO, PO, PO) thin

    comptime _8er_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises -> PO
    comptime _8r_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin -> PO
    comptime _8e_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin raises
    comptime _8_kwargs = def(
        PO, PO, PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) thin

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8er, Self.self_type], f: Self._8er
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8r, Self.self_type], f: Self._8r
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8e, Self.self_type], f: Self._8e
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[Self._8, Self.self_type], f: Self._8
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8er_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8r_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8e_kwargs,
    ):
        self._func = f

    @doc_hidden
    @implicit
    def __init__(
        out self: PyObjectFunction[
            Self._8_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._8_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # Helper utilities
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    def _get_self_arg(
        py_self: PythonObject,
    ) -> UnsafePointer[Self.self_type, MutAnyOrigin]:
        """Get the appropriate self argument for method calls with automatic downcasting.

        Args:
            py_self: The Python object representing self.

        Returns:
            The self argument to pass to the method - downcasted pointer.

        Note:
            This function will abort if downcasting fails for non-PythonObject types.
        """

        comptime assert not _type_is_eq[
            Self.self_type, NoneType
        ](), "Cannot get self arg for NoneType"
        try:
            return py_self.downcast_value_ptr[Self.self_type]()
        except e:
            abort(
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type: "
                    ),
                    e,
                )
            )

    @staticmethod
    def _convert_kwargs(
        py_kwargs: PythonObject,
    ) raises -> OwnedKwargsDict[PythonObject]:
        """Convert a Python dictionary to an OwnedKwargsDict.

        Args:
            py_kwargs: Python dictionary containing keyword arguments.

        Returns:
            An OwnedKwargsDict containing the keyword arguments.
        """
        var result = OwnedKwargsDict[PythonObject]()

        # Handle the case where kwargs is None or empty
        if not py_kwargs._obj_ptr:
            return result^

        # Iterate through the Python dictionary and populate OwnedKwargsDict
        var items = py_kwargs.items()
        for item in items:
            var key = item[0]
            var value = item[1]
            var key_str = String(key)
            result[key_str] = value

        return result^

    # ===-------------------------------------------------------------------===#
    # Compile-time check utilities
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    def _has_type[other_func_type: TrivialRegisterPassable]() -> Bool:
        return _type_is_eq[Self.func_type, other_func_type]()

    @staticmethod
    @always_inline("nodebug")
    def _has_arity(arity: Int) -> Bool:
        comptime if (
            Self._has_type[Self._0er]()
            or Self._has_type[Self._0r]()
            or Self._has_type[Self._0e]()
            or Self._has_type[Self._0]()
            or Self._has_type[Self._0er_kwargs]()
            or Self._has_type[Self._0r_kwargs]()
            or Self._has_type[Self._0e_kwargs]()
            or Self._has_type[Self._0_kwargs]()
        ):
            return arity == 0
        elif (
            Self._has_type[Self._1er]()
            or Self._has_type[Self._1r]()
            or Self._has_type[Self._1e]()
            or Self._has_type[Self._1]()
            or Self._has_type[Self._1er_self]()
            or Self._has_type[Self._1r_self]()
            or Self._has_type[Self._1e_self]()
            or Self._has_type[Self._1_self]()
            or Self._has_type[Self._1er_kwargs]()
            or Self._has_type[Self._1r_kwargs]()
            or Self._has_type[Self._1e_kwargs]()
            or Self._has_type[Self._1_kwargs]()
            or Self._has_type[Self._1er_self_kwargs]()
            or Self._has_type[Self._1r_self_kwargs]()
            or Self._has_type[Self._1e_self_kwargs]()
            or Self._has_type[Self._1_self_kwargs]()
        ):
            return arity == 1
        elif (
            Self._has_type[Self._2er]()
            or Self._has_type[Self._2r]()
            or Self._has_type[Self._2e]()
            or Self._has_type[Self._2]()
            or Self._has_type[Self._2er_self]()
            or Self._has_type[Self._2r_self]()
            or Self._has_type[Self._2e_self]()
            or Self._has_type[Self._2_self]()
            or Self._has_type[Self._2er_kwargs]()
            or Self._has_type[Self._2r_kwargs]()
            or Self._has_type[Self._2e_kwargs]()
            or Self._has_type[Self._2_kwargs]()
            or Self._has_type[Self._2er_self_kwargs]()
            or Self._has_type[Self._2r_self_kwargs]()
            or Self._has_type[Self._2e_self_kwargs]()
            or Self._has_type[Self._2_self_kwargs]()
        ):
            return arity == 2
        elif (
            Self._has_type[Self._3er]()
            or Self._has_type[Self._3r]()
            or Self._has_type[Self._3e]()
            or Self._has_type[Self._3]()
            or Self._has_type[Self._3er_self]()
            or Self._has_type[Self._3r_self]()
            or Self._has_type[Self._3e_self]()
            or Self._has_type[Self._3_self]()
            or Self._has_type[Self._3er_kwargs]()
            or Self._has_type[Self._3r_kwargs]()
            or Self._has_type[Self._3e_kwargs]()
            or Self._has_type[Self._3_kwargs]()
            or Self._has_type[Self._3er_self_kwargs]()
            or Self._has_type[Self._3r_self_kwargs]()
            or Self._has_type[Self._3e_self_kwargs]()
            or Self._has_type[Self._3_self_kwargs]()
        ):
            return arity == 3
        elif (
            Self._has_type[Self._4er]()
            or Self._has_type[Self._4r]()
            or Self._has_type[Self._4e]()
            or Self._has_type[Self._4]()
            or Self._has_type[Self._4er_self]()
            or Self._has_type[Self._4r_self]()
            or Self._has_type[Self._4e_self]()
            or Self._has_type[Self._4_self]()
            or Self._has_type[Self._4er_kwargs]()
            or Self._has_type[Self._4r_kwargs]()
            or Self._has_type[Self._4e_kwargs]()
            or Self._has_type[Self._4_kwargs]()
            or Self._has_type[Self._4er_self_kwargs]()
            or Self._has_type[Self._4r_self_kwargs]()
            or Self._has_type[Self._4e_self_kwargs]()
            or Self._has_type[Self._4_self_kwargs]()
        ):
            return arity == 4
        elif (
            Self._has_type[Self._5er]()
            or Self._has_type[Self._5r]()
            or Self._has_type[Self._5e]()
            or Self._has_type[Self._5]()
            or Self._has_type[Self._5er_self]()
            or Self._has_type[Self._5r_self]()
            or Self._has_type[Self._5e_self]()
            or Self._has_type[Self._5_self]()
            or Self._has_type[Self._5er_kwargs]()
            or Self._has_type[Self._5r_kwargs]()
            or Self._has_type[Self._5e_kwargs]()
            or Self._has_type[Self._5_kwargs]()
            or Self._has_type[Self._5er_self_kwargs]()
            or Self._has_type[Self._5r_self_kwargs]()
            or Self._has_type[Self._5e_self_kwargs]()
            or Self._has_type[Self._5_self_kwargs]()
        ):
            return arity == 5
        elif (
            Self._has_type[Self._6er]()
            or Self._has_type[Self._6r]()
            or Self._has_type[Self._6e]()
            or Self._has_type[Self._6]()
            or Self._has_type[Self._6er_self]()
            or Self._has_type[Self._6r_self]()
            or Self._has_type[Self._6e_self]()
            or Self._has_type[Self._6_self]()
            or Self._has_type[Self._6er_kwargs]()
            or Self._has_type[Self._6r_kwargs]()
            or Self._has_type[Self._6e_kwargs]()
            or Self._has_type[Self._6_kwargs]()
            or Self._has_type[Self._6er_self_kwargs]()
            or Self._has_type[Self._6r_self_kwargs]()
            or Self._has_type[Self._6e_self_kwargs]()
            or Self._has_type[Self._6_self_kwargs]()
        ):
            return arity == 6
        elif (
            Self._has_type[Self._7er]()
            or Self._has_type[Self._7r]()
            or Self._has_type[Self._7e]()
            or Self._has_type[Self._7]()
            or Self._has_type[Self._7er_self]()
            or Self._has_type[Self._7r_self]()
            or Self._has_type[Self._7e_self]()
            or Self._has_type[Self._7_self]()
            or Self._has_type[Self._7er_kwargs]()
            or Self._has_type[Self._7r_kwargs]()
            or Self._has_type[Self._7e_kwargs]()
            or Self._has_type[Self._7_kwargs]()
            or Self._has_type[Self._7er_self_kwargs]()
            or Self._has_type[Self._7r_self_kwargs]()
            or Self._has_type[Self._7e_self_kwargs]()
            or Self._has_type[Self._7_self_kwargs]()
        ):
            return arity == 7
        elif (
            Self._has_type[Self._8er]()
            or Self._has_type[Self._8r]()
            or Self._has_type[Self._8e]()
            or Self._has_type[Self._8]()
            or Self._has_type[Self._8er_self]()
            or Self._has_type[Self._8r_self]()
            or Self._has_type[Self._8e_self]()
            or Self._has_type[Self._8_self]()
            or Self._has_type[Self._8er_kwargs]()
            or Self._has_type[Self._8r_kwargs]()
            or Self._has_type[Self._8e_kwargs]()
            or Self._has_type[Self._8_kwargs]()
            or Self._has_type[Self._8er_self_kwargs]()
            or Self._has_type[Self._8r_self_kwargs]()
            or Self._has_type[Self._8e_self_kwargs]()
            or Self._has_type[Self._8_self_kwargs]()
        ):
            return arity == 8
        else:
            return False

    # ===-------------------------------------------------------------------===#
    # Compile-time dispatch
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    def _dispatch[
        is_method: Bool,
    ](func: Self.func_type, py_self: PO, py_args: PO) raises -> PO:
        """Compile-time dispatch for non-kwargs function/method calls."""

        comptime if Self._has_arity(0):
            comptime assert not is_method, "arity-0 methods not supported"
            check_arguments_arity(0, py_args)
            comptime if Self._has_type[Self._0er]():
                return rebind[Self._0er](func)()
            elif Self._has_type[Self._0r]():
                return rebind[Self._0r](func)()
            elif Self._has_type[Self._0e]():
                rebind[Self._0e](func)()
                return PO(None)
            else:
                rebind[Self._0](func)()
                return PO(None)
        elif Self._has_arity(1):
            comptime if not is_method:
                check_arguments_arity(1, py_args)
                var a0 = py_args[0]
                comptime if Self._has_type[Self._1er]():
                    return rebind[Self._1er](func)(a0)
                elif Self._has_type[Self._1r]():
                    return rebind[Self._1r](func)(a0)
                elif Self._has_type[Self._1e]():
                    rebind[Self._1e](func)(a0)
                    return PO(None)
                else:
                    rebind[Self._1](func)(a0)
                    return PO(None)
            else:
                check_arguments_arity(0, py_args)
                comptime if Self._has_type[Self._1er]():
                    return rebind[Self._1er](func)(py_self)
                elif Self._has_type[Self._1r]():
                    return rebind[Self._1r](func)(py_self)
                elif Self._has_type[Self._1e]():
                    rebind[Self._1e](func)(py_self)
                    return PO(None)
                elif Self._has_type[Self._1]():
                    rebind[Self._1](func)(py_self)
                    return PO(None)
                elif Self._has_type[Self._1er_self]():
                    return rebind[Self._1er_self](func)(
                        Self._get_self_arg(py_self)
                    )
                elif Self._has_type[Self._1r_self]():
                    return rebind[Self._1r_self](func)(
                        Self._get_self_arg(py_self)
                    )
                elif Self._has_type[Self._1e_self]():
                    rebind[Self._1e_self](func)(Self._get_self_arg(py_self))
                    return PO(None)
                else:
                    rebind[Self._1_self](func)(Self._get_self_arg(py_self))
                    return PO(None)
        elif Self._has_arity(2):
            comptime if not is_method:
                check_arguments_arity(2, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                comptime if Self._has_type[Self._2er]():
                    return rebind[Self._2er](func)(a0, a1)
                elif Self._has_type[Self._2r]():
                    return rebind[Self._2r](func)(a0, a1)
                elif Self._has_type[Self._2e]():
                    rebind[Self._2e](func)(a0, a1)
                    return PO(None)
                else:
                    rebind[Self._2](func)(a0, a1)
                    return PO(None)
            else:
                check_arguments_arity(1, py_args)
                var a0 = py_args[0]
                comptime if Self._has_type[Self._2er]():
                    return rebind[Self._2er](func)(py_self, a0)
                elif Self._has_type[Self._2r]():
                    return rebind[Self._2r](func)(py_self, a0)
                elif Self._has_type[Self._2e]():
                    rebind[Self._2e](func)(py_self, a0)
                    return PO(None)
                elif Self._has_type[Self._2]():
                    rebind[Self._2](func)(py_self, a0)
                    return PO(None)
                elif Self._has_type[Self._2er_self]():
                    return rebind[Self._2er_self](func)(
                        Self._get_self_arg(py_self), a0
                    )
                elif Self._has_type[Self._2r_self]():
                    return rebind[Self._2r_self](func)(
                        Self._get_self_arg(py_self), a0
                    )
                elif Self._has_type[Self._2e_self]():
                    rebind[Self._2e_self](func)(Self._get_self_arg(py_self), a0)
                    return PO(None)
                else:
                    rebind[Self._2_self](func)(Self._get_self_arg(py_self), a0)
                    return PO(None)
        elif Self._has_arity(3):
            comptime if not is_method:
                check_arguments_arity(3, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                comptime if Self._has_type[Self._3er]():
                    return rebind[Self._3er](func)(a0, a1, a2)
                elif Self._has_type[Self._3r]():
                    return rebind[Self._3r](func)(a0, a1, a2)
                elif Self._has_type[Self._3e]():
                    rebind[Self._3e](func)(a0, a1, a2)
                    return PO(None)
                else:
                    rebind[Self._3](func)(a0, a1, a2)
                    return PO(None)
            else:
                check_arguments_arity(2, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                comptime if Self._has_type[Self._3er]():
                    return rebind[Self._3er](func)(py_self, a0, a1)
                elif Self._has_type[Self._3r]():
                    return rebind[Self._3r](func)(py_self, a0, a1)
                elif Self._has_type[Self._3e]():
                    rebind[Self._3e](func)(py_self, a0, a1)
                    return PO(None)
                elif Self._has_type[Self._3]():
                    rebind[Self._3](func)(py_self, a0, a1)
                    return PO(None)
                elif Self._has_type[Self._3er_self]():
                    return rebind[Self._3er_self](func)(
                        Self._get_self_arg(py_self), a0, a1
                    )
                elif Self._has_type[Self._3r_self]():
                    return rebind[Self._3r_self](func)(
                        Self._get_self_arg(py_self), a0, a1
                    )
                elif Self._has_type[Self._3e_self]():
                    rebind[Self._3e_self](func)(
                        Self._get_self_arg(py_self), a0, a1
                    )
                    return PO(None)
                else:
                    rebind[Self._3_self](func)(
                        Self._get_self_arg(py_self), a0, a1
                    )
                    return PO(None)
        elif Self._has_arity(4):
            comptime if not is_method:
                check_arguments_arity(4, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                comptime if Self._has_type[Self._4er]():
                    return rebind[Self._4er](func)(a0, a1, a2, a3)
                elif Self._has_type[Self._4r]():
                    return rebind[Self._4r](func)(a0, a1, a2, a3)
                elif Self._has_type[Self._4e]():
                    rebind[Self._4e](func)(a0, a1, a2, a3)
                    return PO(None)
                else:
                    rebind[Self._4](func)(a0, a1, a2, a3)
                    return PO(None)
            else:
                check_arguments_arity(3, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                comptime if Self._has_type[Self._4er]():
                    return rebind[Self._4er](func)(py_self, a0, a1, a2)
                elif Self._has_type[Self._4r]():
                    return rebind[Self._4r](func)(py_self, a0, a1, a2)
                elif Self._has_type[Self._4e]():
                    rebind[Self._4e](func)(py_self, a0, a1, a2)
                    return PO(None)
                elif Self._has_type[Self._4]():
                    rebind[Self._4](func)(py_self, a0, a1, a2)
                    return PO(None)
                elif Self._has_type[Self._4er_self]():
                    return rebind[Self._4er_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2
                    )
                elif Self._has_type[Self._4r_self]():
                    return rebind[Self._4r_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2
                    )
                elif Self._has_type[Self._4e_self]():
                    rebind[Self._4e_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2
                    )
                    return PO(None)
                else:
                    rebind[Self._4_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2
                    )
                    return PO(None)
        elif Self._has_arity(5):
            comptime if not is_method:
                check_arguments_arity(5, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                comptime if Self._has_type[Self._5er]():
                    return rebind[Self._5er](func)(a0, a1, a2, a3, a4)
                elif Self._has_type[Self._5r]():
                    return rebind[Self._5r](func)(a0, a1, a2, a3, a4)
                elif Self._has_type[Self._5e]():
                    rebind[Self._5e](func)(a0, a1, a2, a3, a4)
                    return PO(None)
                else:
                    rebind[Self._5](func)(a0, a1, a2, a3, a4)
                    return PO(None)
            else:
                check_arguments_arity(4, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                comptime if Self._has_type[Self._5er]():
                    return rebind[Self._5er](func)(py_self, a0, a1, a2, a3)
                elif Self._has_type[Self._5r]():
                    return rebind[Self._5r](func)(py_self, a0, a1, a2, a3)
                elif Self._has_type[Self._5e]():
                    rebind[Self._5e](func)(py_self, a0, a1, a2, a3)
                    return PO(None)
                elif Self._has_type[Self._5]():
                    rebind[Self._5](func)(py_self, a0, a1, a2, a3)
                    return PO(None)
                elif Self._has_type[Self._5er_self]():
                    return rebind[Self._5er_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3
                    )
                elif Self._has_type[Self._5r_self]():
                    return rebind[Self._5r_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3
                    )
                elif Self._has_type[Self._5e_self]():
                    rebind[Self._5e_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3
                    )
                    return PO(None)
                else:
                    rebind[Self._5_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3
                    )
                    return PO(None)
        elif Self._has_arity(6):
            comptime if not is_method:
                check_arguments_arity(6, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                comptime if Self._has_type[Self._6er]():
                    return rebind[Self._6er](func)(a0, a1, a2, a3, a4, a5)
                elif Self._has_type[Self._6r]():
                    return rebind[Self._6r](func)(a0, a1, a2, a3, a4, a5)
                elif Self._has_type[Self._6e]():
                    rebind[Self._6e](func)(a0, a1, a2, a3, a4, a5)
                    return PO(None)
                else:
                    rebind[Self._6](func)(a0, a1, a2, a3, a4, a5)
                    return PO(None)
            else:
                check_arguments_arity(5, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                comptime if Self._has_type[Self._6er]():
                    return rebind[Self._6er](func)(py_self, a0, a1, a2, a3, a4)
                elif Self._has_type[Self._6r]():
                    return rebind[Self._6r](func)(py_self, a0, a1, a2, a3, a4)
                elif Self._has_type[Self._6e]():
                    rebind[Self._6e](func)(py_self, a0, a1, a2, a3, a4)
                    return PO(None)
                elif Self._has_type[Self._6]():
                    rebind[Self._6](func)(py_self, a0, a1, a2, a3, a4)
                    return PO(None)
                elif Self._has_type[Self._6er_self]():
                    return rebind[Self._6er_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4
                    )
                elif Self._has_type[Self._6r_self]():
                    return rebind[Self._6r_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4
                    )
                elif Self._has_type[Self._6e_self]():
                    rebind[Self._6e_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4
                    )
                    return PO(None)
                else:
                    rebind[Self._6_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4
                    )
                    return PO(None)
        elif Self._has_arity(7):
            comptime if not is_method:
                check_arguments_arity(7, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                var a6 = py_args[6]
                comptime if Self._has_type[Self._7er]():
                    return rebind[Self._7er](func)(a0, a1, a2, a3, a4, a5, a6)
                elif Self._has_type[Self._7r]():
                    return rebind[Self._7r](func)(a0, a1, a2, a3, a4, a5, a6)
                elif Self._has_type[Self._7e]():
                    rebind[Self._7e](func)(a0, a1, a2, a3, a4, a5, a6)
                    return PO(None)
                else:
                    rebind[Self._7](func)(a0, a1, a2, a3, a4, a5, a6)
                    return PO(None)
            else:
                check_arguments_arity(6, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                comptime if Self._has_type[Self._7er]():
                    return rebind[Self._7er](func)(
                        py_self, a0, a1, a2, a3, a4, a5
                    )
                elif Self._has_type[Self._7r]():
                    return rebind[Self._7r](func)(
                        py_self, a0, a1, a2, a3, a4, a5
                    )
                elif Self._has_type[Self._7e]():
                    rebind[Self._7e](func)(py_self, a0, a1, a2, a3, a4, a5)
                    return PO(None)
                elif Self._has_type[Self._7]():
                    rebind[Self._7](func)(py_self, a0, a1, a2, a3, a4, a5)
                    return PO(None)
                elif Self._has_type[Self._7er_self]():
                    return rebind[Self._7er_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5
                    )
                elif Self._has_type[Self._7r_self]():
                    return rebind[Self._7r_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5
                    )
                elif Self._has_type[Self._7e_self]():
                    rebind[Self._7e_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5
                    )
                    return PO(None)
                else:
                    rebind[Self._7_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5
                    )
                    return PO(None)
        elif Self._has_arity(8):
            comptime if not is_method:
                check_arguments_arity(8, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                var a6 = py_args[6]
                var a7 = py_args[7]
                comptime if Self._has_type[Self._8er]():
                    return rebind[Self._8er](func)(
                        a0, a1, a2, a3, a4, a5, a6, a7
                    )
                elif Self._has_type[Self._8r]():
                    return rebind[Self._8r](func)(
                        a0, a1, a2, a3, a4, a5, a6, a7
                    )
                elif Self._has_type[Self._8e]():
                    rebind[Self._8e](func)(a0, a1, a2, a3, a4, a5, a6, a7)
                    return PO(None)
                else:
                    rebind[Self._8](func)(a0, a1, a2, a3, a4, a5, a6, a7)
                    return PO(None)
            else:
                check_arguments_arity(7, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                var a6 = py_args[6]
                comptime if Self._has_type[Self._8er]():
                    return rebind[Self._8er](func)(
                        py_self, a0, a1, a2, a3, a4, a5, a6
                    )
                elif Self._has_type[Self._8r]():
                    return rebind[Self._8r](func)(
                        py_self, a0, a1, a2, a3, a4, a5, a6
                    )
                elif Self._has_type[Self._8e]():
                    rebind[Self._8e](func)(py_self, a0, a1, a2, a3, a4, a5, a6)
                    return PO(None)
                elif Self._has_type[Self._8]():
                    rebind[Self._8](func)(py_self, a0, a1, a2, a3, a4, a5, a6)
                    return PO(None)
                elif Self._has_type[Self._8er_self]():
                    return rebind[Self._8er_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5, a6
                    )
                elif Self._has_type[Self._8r_self]():
                    return rebind[Self._8r_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5, a6
                    )
                elif Self._has_type[Self._8e_self]():
                    rebind[Self._8e_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5, a6
                    )
                    return PO(None)
                else:
                    rebind[Self._8_self](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, a5, a6
                    )
                    return PO(None)
        else:
            comptime assert False, "unsupported arity"

    @staticmethod
    @always_inline("nodebug")
    def _dispatch_kwargs[
        is_method: Bool,
    ](
        func: Self.func_type,
        py_self: PO,
        py_args: PO,
        kwargs: OwnedKwargsDict[PO],
    ) raises -> PO:
        """Compile-time dispatch for kwargs function/method calls."""
        comptime assert Self.has_kwargs, "only for kwargs functions"

        comptime if Self._has_arity(0):
            comptime assert not is_method, "arity-0 methods not supported"
            check_arguments_arity(0, py_args)
            comptime if Self._has_type[Self._0er_kwargs]():
                return rebind[Self._0er_kwargs](func)(kwargs)
            elif Self._has_type[Self._0r_kwargs]():
                return rebind[Self._0r_kwargs](func)(kwargs)
            elif Self._has_type[Self._0e_kwargs]():
                rebind[Self._0e_kwargs](func)(kwargs)
                return PO(None)
            else:
                rebind[Self._0_kwargs](func)(kwargs)
                return PO(None)
        elif Self._has_arity(1):
            comptime if not is_method:
                check_arguments_arity(1, py_args)
                var a0 = py_args[0]
                comptime if Self._has_type[Self._1er_kwargs]():
                    return rebind[Self._1er_kwargs](func)(a0, kwargs)
                elif Self._has_type[Self._1r_kwargs]():
                    return rebind[Self._1r_kwargs](func)(a0, kwargs)
                elif Self._has_type[Self._1e_kwargs]():
                    rebind[Self._1e_kwargs](func)(a0, kwargs)
                    return PO(None)
                else:
                    rebind[Self._1_kwargs](func)(a0, kwargs)
                    return PO(None)
            else:
                check_arguments_arity(0, py_args)
                comptime if Self._has_type[Self._1er_kwargs]():
                    return rebind[Self._1er_kwargs](func)(py_self, kwargs)
                elif Self._has_type[Self._1r_kwargs]():
                    return rebind[Self._1r_kwargs](func)(py_self, kwargs)
                elif Self._has_type[Self._1e_kwargs]():
                    rebind[Self._1e_kwargs](func)(py_self, kwargs)
                    return PO(None)
                elif Self._has_type[Self._1_kwargs]():
                    rebind[Self._1_kwargs](func)(py_self, kwargs)
                    return PO(None)
                elif Self._has_type[Self._1er_self_kwargs]():
                    return rebind[Self._1er_self_kwargs](func)(
                        Self._get_self_arg(py_self), kwargs
                    )
                elif Self._has_type[Self._1r_self_kwargs]():
                    return rebind[Self._1r_self_kwargs](func)(
                        Self._get_self_arg(py_self), kwargs
                    )
                elif Self._has_type[Self._1e_self_kwargs]():
                    rebind[Self._1e_self_kwargs](func)(
                        Self._get_self_arg(py_self), kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._1_self_kwargs](func)(
                        Self._get_self_arg(py_self), kwargs
                    )
                    return PO(None)
        elif Self._has_arity(2):
            comptime if not is_method:
                check_arguments_arity(2, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                comptime if Self._has_type[Self._2er_kwargs]():
                    return rebind[Self._2er_kwargs](func)(a0, a1, kwargs)
                elif Self._has_type[Self._2r_kwargs]():
                    return rebind[Self._2r_kwargs](func)(a0, a1, kwargs)
                elif Self._has_type[Self._2e_kwargs]():
                    rebind[Self._2e_kwargs](func)(a0, a1, kwargs)
                    return PO(None)
                else:
                    rebind[Self._2_kwargs](func)(a0, a1, kwargs)
                    return PO(None)
            else:
                check_arguments_arity(1, py_args)
                var a0 = py_args[0]
                comptime if Self._has_type[Self._2er_kwargs]():
                    return rebind[Self._2er_kwargs](func)(py_self, a0, kwargs)
                elif Self._has_type[Self._2r_kwargs]():
                    return rebind[Self._2r_kwargs](func)(py_self, a0, kwargs)
                elif Self._has_type[Self._2e_kwargs]():
                    rebind[Self._2e_kwargs](func)(py_self, a0, kwargs)
                    return PO(None)
                elif Self._has_type[Self._2_kwargs]():
                    rebind[Self._2_kwargs](func)(py_self, a0, kwargs)
                    return PO(None)
                elif Self._has_type[Self._2er_self_kwargs]():
                    return rebind[Self._2er_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, kwargs
                    )
                elif Self._has_type[Self._2r_self_kwargs]():
                    return rebind[Self._2r_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, kwargs
                    )
                elif Self._has_type[Self._2e_self_kwargs]():
                    rebind[Self._2e_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._2_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, kwargs
                    )
                    return PO(None)
        elif Self._has_arity(3):
            comptime if not is_method:
                check_arguments_arity(3, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                comptime if Self._has_type[Self._3er_kwargs]():
                    return rebind[Self._3er_kwargs](func)(a0, a1, a2, kwargs)
                elif Self._has_type[Self._3r_kwargs]():
                    return rebind[Self._3r_kwargs](func)(a0, a1, a2, kwargs)
                elif Self._has_type[Self._3e_kwargs]():
                    rebind[Self._3e_kwargs](func)(a0, a1, a2, kwargs)
                    return PO(None)
                else:
                    rebind[Self._3_kwargs](func)(a0, a1, a2, kwargs)
                    return PO(None)
            else:
                check_arguments_arity(2, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                comptime if Self._has_type[Self._3er_kwargs]():
                    return rebind[Self._3er_kwargs](func)(
                        py_self, a0, a1, kwargs
                    )
                elif Self._has_type[Self._3r_kwargs]():
                    return rebind[Self._3r_kwargs](func)(
                        py_self, a0, a1, kwargs
                    )
                elif Self._has_type[Self._3e_kwargs]():
                    rebind[Self._3e_kwargs](func)(py_self, a0, a1, kwargs)
                    return PO(None)
                elif Self._has_type[Self._3_kwargs]():
                    rebind[Self._3_kwargs](func)(py_self, a0, a1, kwargs)
                    return PO(None)
                elif Self._has_type[Self._3er_self_kwargs]():
                    return rebind[Self._3er_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, kwargs
                    )
                elif Self._has_type[Self._3r_self_kwargs]():
                    return rebind[Self._3r_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, kwargs
                    )
                elif Self._has_type[Self._3e_self_kwargs]():
                    rebind[Self._3e_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._3_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, kwargs
                    )
                    return PO(None)
        elif Self._has_arity(4):
            comptime if not is_method:
                check_arguments_arity(4, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                comptime if Self._has_type[Self._4er_kwargs]():
                    return rebind[Self._4er_kwargs](func)(
                        a0, a1, a2, a3, kwargs
                    )
                elif Self._has_type[Self._4r_kwargs]():
                    return rebind[Self._4r_kwargs](func)(a0, a1, a2, a3, kwargs)
                elif Self._has_type[Self._4e_kwargs]():
                    rebind[Self._4e_kwargs](func)(a0, a1, a2, a3, kwargs)
                    return PO(None)
                else:
                    rebind[Self._4_kwargs](func)(a0, a1, a2, a3, kwargs)
                    return PO(None)
            else:
                check_arguments_arity(3, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                comptime if Self._has_type[Self._4er_kwargs]():
                    return rebind[Self._4er_kwargs](func)(
                        py_self, a0, a1, a2, kwargs
                    )
                elif Self._has_type[Self._4r_kwargs]():
                    return rebind[Self._4r_kwargs](func)(
                        py_self, a0, a1, a2, kwargs
                    )
                elif Self._has_type[Self._4e_kwargs]():
                    rebind[Self._4e_kwargs](func)(py_self, a0, a1, a2, kwargs)
                    return PO(None)
                elif Self._has_type[Self._4_kwargs]():
                    rebind[Self._4_kwargs](func)(py_self, a0, a1, a2, kwargs)
                    return PO(None)
                elif Self._has_type[Self._4er_self_kwargs]():
                    return rebind[Self._4er_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, kwargs
                    )
                elif Self._has_type[Self._4r_self_kwargs]():
                    return rebind[Self._4r_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, kwargs
                    )
                elif Self._has_type[Self._4e_self_kwargs]():
                    rebind[Self._4e_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._4_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, kwargs
                    )
                    return PO(None)
        elif Self._has_arity(5):
            comptime if not is_method:
                check_arguments_arity(5, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                comptime if Self._has_type[Self._5er_kwargs]():
                    return rebind[Self._5er_kwargs](func)(
                        a0, a1, a2, a3, a4, kwargs
                    )
                elif Self._has_type[Self._5r_kwargs]():
                    return rebind[Self._5r_kwargs](func)(
                        a0, a1, a2, a3, a4, kwargs
                    )
                elif Self._has_type[Self._5e_kwargs]():
                    rebind[Self._5e_kwargs](func)(a0, a1, a2, a3, a4, kwargs)
                    return PO(None)
                else:
                    rebind[Self._5_kwargs](func)(a0, a1, a2, a3, a4, kwargs)
                    return PO(None)
            else:
                check_arguments_arity(4, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                comptime if Self._has_type[Self._5er_kwargs]():
                    return rebind[Self._5er_kwargs](func)(
                        py_self, a0, a1, a2, a3, kwargs
                    )
                elif Self._has_type[Self._5r_kwargs]():
                    return rebind[Self._5r_kwargs](func)(
                        py_self, a0, a1, a2, a3, kwargs
                    )
                elif Self._has_type[Self._5e_kwargs]():
                    rebind[Self._5e_kwargs](func)(
                        py_self, a0, a1, a2, a3, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._5_kwargs]():
                    rebind[Self._5_kwargs](func)(
                        py_self, a0, a1, a2, a3, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._5er_self_kwargs]():
                    return rebind[Self._5er_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, kwargs
                    )
                elif Self._has_type[Self._5r_self_kwargs]():
                    return rebind[Self._5r_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, kwargs
                    )
                elif Self._has_type[Self._5e_self_kwargs]():
                    rebind[Self._5e_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._5_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, kwargs
                    )
                    return PO(None)
        elif Self._has_arity(6):
            comptime if not is_method:
                check_arguments_arity(6, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                comptime if Self._has_type[Self._6er_kwargs]():
                    return rebind[Self._6er_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, kwargs
                    )
                elif Self._has_type[Self._6r_kwargs]():
                    return rebind[Self._6r_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, kwargs
                    )
                elif Self._has_type[Self._6e_kwargs]():
                    rebind[Self._6e_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._6_kwargs](func)(a0, a1, a2, a3, a4, a5, kwargs)
                    return PO(None)
            else:
                check_arguments_arity(5, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                comptime if Self._has_type[Self._6er_kwargs]():
                    return rebind[Self._6er_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, kwargs
                    )
                elif Self._has_type[Self._6r_kwargs]():
                    return rebind[Self._6r_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, kwargs
                    )
                elif Self._has_type[Self._6e_kwargs]():
                    rebind[Self._6e_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._6_kwargs]():
                    rebind[Self._6_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._6er_self_kwargs]():
                    return rebind[Self._6er_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, kwargs
                    )
                elif Self._has_type[Self._6r_self_kwargs]():
                    return rebind[Self._6r_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, kwargs
                    )
                elif Self._has_type[Self._6e_self_kwargs]():
                    rebind[Self._6e_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._6_self_kwargs](func)(
                        Self._get_self_arg(py_self), a0, a1, a2, a3, a4, kwargs
                    )
                    return PO(None)
        elif Self._has_arity(7):
            comptime if not is_method:
                check_arguments_arity(7, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                var a6 = py_args[6]
                comptime if Self._has_type[Self._7er_kwargs]():
                    return rebind[Self._7er_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                elif Self._has_type[Self._7r_kwargs]():
                    return rebind[Self._7r_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                elif Self._has_type[Self._7e_kwargs]():
                    rebind[Self._7e_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._7_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                    return PO(None)
            else:
                check_arguments_arity(6, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                comptime if Self._has_type[Self._7er_kwargs]():
                    return rebind[Self._7er_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, kwargs
                    )
                elif Self._has_type[Self._7r_kwargs]():
                    return rebind[Self._7r_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, kwargs
                    )
                elif Self._has_type[Self._7e_kwargs]():
                    rebind[Self._7e_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._7_kwargs]():
                    rebind[Self._7_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._7er_self_kwargs]():
                    return rebind[Self._7er_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        kwargs,
                    )
                elif Self._has_type[Self._7r_self_kwargs]():
                    return rebind[Self._7r_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        kwargs,
                    )
                elif Self._has_type[Self._7e_self_kwargs]():
                    rebind[Self._7e_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        kwargs,
                    )
                    return PO(None)
                else:
                    rebind[Self._7_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        kwargs,
                    )
                    return PO(None)
        elif Self._has_arity(8):
            comptime if not is_method:
                check_arguments_arity(8, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                var a6 = py_args[6]
                var a7 = py_args[7]
                comptime if Self._has_type[Self._8er_kwargs]():
                    return rebind[Self._8er_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, a7, kwargs
                    )
                elif Self._has_type[Self._8r_kwargs]():
                    return rebind[Self._8r_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, a7, kwargs
                    )
                elif Self._has_type[Self._8e_kwargs]():
                    rebind[Self._8e_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, a7, kwargs
                    )
                    return PO(None)
                else:
                    rebind[Self._8_kwargs](func)(
                        a0, a1, a2, a3, a4, a5, a6, a7, kwargs
                    )
                    return PO(None)
            else:
                check_arguments_arity(7, py_args)
                var a0 = py_args[0]
                var a1 = py_args[1]
                var a2 = py_args[2]
                var a3 = py_args[3]
                var a4 = py_args[4]
                var a5 = py_args[5]
                var a6 = py_args[6]
                comptime if Self._has_type[Self._8er_kwargs]():
                    return rebind[Self._8er_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                elif Self._has_type[Self._8r_kwargs]():
                    return rebind[Self._8r_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                elif Self._has_type[Self._8e_kwargs]():
                    rebind[Self._8e_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._8_kwargs]():
                    rebind[Self._8_kwargs](func)(
                        py_self, a0, a1, a2, a3, a4, a5, a6, kwargs
                    )
                    return PO(None)
                elif Self._has_type[Self._8er_self_kwargs]():
                    return rebind[Self._8er_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        a6,
                        kwargs,
                    )
                elif Self._has_type[Self._8r_self_kwargs]():
                    return rebind[Self._8r_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        a6,
                        kwargs,
                    )
                elif Self._has_type[Self._8e_self_kwargs]():
                    rebind[Self._8e_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        a6,
                        kwargs,
                    )
                    return PO(None)
                else:
                    rebind[Self._8_self_kwargs](func)(
                        Self._get_self_arg(py_self),
                        a0,
                        a1,
                        a2,
                        a3,
                        a4,
                        a5,
                        a6,
                        kwargs,
                    )
                    return PO(None)
        else:
            comptime assert False, "unsupported arity"
