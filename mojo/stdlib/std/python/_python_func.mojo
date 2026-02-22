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

from collections import OwnedKwargsDict
from os import abort
from sys.intrinsics import _type_is_eq

from python import PythonObject as PO  # for brevity of signatures below
from python.bindings import check_arguments_arity


struct PyObjectFunction[
    func_type: __TypeOfAllTypes,
    self_type: ImplicitlyDestructible = NoneType,
    has_kwargs: Bool = False,
](ImplicitlyCopyable):
    """Wrapper to hide the binding logic for functions taking a variadic number
    of PythonObject arguments.

    This currently supports function types with up to 6 positional arguments,
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

    comptime _0er = fn() raises -> PO
    comptime _0r = fn() -> PO
    comptime _0e = fn() raises
    comptime _0 = fn()

    comptime _0er_kwargs = fn(OwnedKwargsDict[PO]) raises -> PO
    comptime _0r_kwargs = fn(OwnedKwargsDict[PO]) -> PO
    comptime _0e_kwargs = fn(OwnedKwargsDict[PO]) raises
    comptime _0_kwargs = fn(OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0er], f: Self._0er):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0r], f: Self._0r):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0e], f: Self._0e):
        self._func = f

    @doc_private
    @implicit
    fn __init__(out self: PyObjectFunction[Self._0], f: Self._0):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._0er_kwargs, has_kwargs=True],
        f: Self._0er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._0r_kwargs, has_kwargs=True],
        f: Self._0r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._0e_kwargs, has_kwargs=True],
        f: Self._0e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._0_kwargs, has_kwargs=True],
        f: Self._0_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 1 argument
    # ===-------------------------------------------------------------------===#

    comptime _1er = fn(PO) raises -> PO
    comptime _1r = fn(PO) -> PO
    comptime _1e = fn(PO) raises
    comptime _1 = fn(PO)

    comptime _1er_kwargs = fn(PO, OwnedKwargsDict[PO]) raises -> PO
    comptime _1r_kwargs = fn(PO, OwnedKwargsDict[PO]) -> PO
    comptime _1e_kwargs = fn(PO, OwnedKwargsDict[PO]) raises
    comptime _1_kwargs = fn(PO, OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1er, Self.self_type], f: Self._1er
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1r, Self.self_type], f: Self._1r
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1e, Self.self_type], f: Self._1e
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1, Self.self_type], f: Self._1
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 1 argument (typed self methods - 0 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _1er_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin]
    ) raises -> PO
    comptime _1r_self = fn(UnsafePointer[Self.self_type, MutAnyOrigin]) -> PO
    comptime _1e_self = fn(UnsafePointer[Self.self_type, MutAnyOrigin]) raises
    comptime _1_self = fn(UnsafePointer[Self.self_type, MutAnyOrigin])

    comptime _1er_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) raises -> PO
    comptime _1r_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) -> PO
    comptime _1e_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    ) raises
    comptime _1_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], OwnedKwargsDict[PO]
    )

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1er_self, Self.self_type],
        f: Self._1er_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1r_self, Self.self_type],
        f: Self._1r_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1e_self, Self.self_type],
        f: Self._1e_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._1_self, Self.self_type],
        f: Self._1_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1er_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1r_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1e_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._1_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._1_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 2 arguments (typed self methods - 1 additional argument)
    # ===-------------------------------------------------------------------===#

    comptime _2er_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO
    ) raises -> PO
    comptime _2r_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO
    ) -> PO
    comptime _2e_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO
    ) raises
    comptime _2_self = fn(UnsafePointer[Self.self_type, MutAnyOrigin], PO)

    comptime _2er_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) raises -> PO
    comptime _2r_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) -> PO
    comptime _2e_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    ) raises
    comptime _2_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, OwnedKwargsDict[PO]
    )

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2er_self, Self.self_type],
        f: Self._2er_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2r_self, Self.self_type],
        f: Self._2r_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2e_self, Self.self_type],
        f: Self._2e_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2_self, Self.self_type],
        f: Self._2_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2er_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2r_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2e_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 3 arguments (typed self methods - 2 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _3er_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) raises -> PO
    comptime _3r_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) -> PO
    comptime _3e_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO
    ) raises
    comptime _3_self = fn(UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO)

    comptime _3er_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) raises -> PO
    comptime _3r_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) -> PO
    comptime _3e_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    ) raises
    comptime _3_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, OwnedKwargsDict[PO]
    )

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3er_self, Self.self_type],
        f: Self._3er_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3r_self, Self.self_type],
        f: Self._3r_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3e_self, Self.self_type],
        f: Self._3e_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3_self, Self.self_type],
        f: Self._3_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3er_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3r_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3e_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 4 arguments (typed self methods - 3 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _4er_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) raises -> PO
    comptime _4r_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) -> PO
    comptime _4e_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    ) raises
    comptime _4_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO
    )

    comptime _4er_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) raises -> PO
    comptime _4r_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) -> PO
    comptime _4e_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) raises
    comptime _4_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    )

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4er_self, Self.self_type],
        f: Self._4er_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4r_self, Self.self_type],
        f: Self._4r_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4e_self, Self.self_type],
        f: Self._4e_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4_self, Self.self_type],
        f: Self._4_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4er_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4r_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4e_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 2 arguments
    # ===-------------------------------------------------------------------===#

    comptime _2er = fn(PO, PO) raises -> PO
    comptime _2r = fn(PO, PO) -> PO
    comptime _2e = fn(PO, PO) raises
    comptime _2 = fn(PO, PO)

    comptime _2er_kwargs = fn(PO, PO, OwnedKwargsDict[PO]) raises -> PO
    comptime _2r_kwargs = fn(PO, PO, OwnedKwargsDict[PO]) -> PO
    comptime _2e_kwargs = fn(PO, PO, OwnedKwargsDict[PO]) raises
    comptime _2_kwargs = fn(PO, PO, OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2er, Self.self_type], f: Self._2er
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2r, Self.self_type], f: Self._2r
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2e, Self.self_type], f: Self._2e
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._2, Self.self_type], f: Self._2
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._2_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._2_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 3 arguments
    # ===-------------------------------------------------------------------===#

    comptime _3er = fn(PO, PO, PO) raises -> PO
    comptime _3r = fn(PO, PO, PO) -> PO
    comptime _3e = fn(PO, PO, PO) raises
    comptime _3 = fn(PO, PO, PO)

    comptime _3er_kwargs = fn(PO, PO, PO, OwnedKwargsDict[PO]) raises -> PO
    comptime _3r_kwargs = fn(PO, PO, PO, OwnedKwargsDict[PO]) -> PO
    comptime _3e_kwargs = fn(PO, PO, PO, OwnedKwargsDict[PO]) raises
    comptime _3_kwargs = fn(PO, PO, PO, OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3er, Self.self_type], f: Self._3er
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3r, Self.self_type], f: Self._3r
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3e, Self.self_type], f: Self._3e
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._3, Self.self_type], f: Self._3
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._3_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._3_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 4 arguments
    # ===-------------------------------------------------------------------===#

    comptime _4er = fn(PO, PO, PO, PO) raises -> PO
    comptime _4r = fn(PO, PO, PO, PO) -> PO
    comptime _4e = fn(PO, PO, PO, PO) raises
    comptime _4 = fn(PO, PO, PO, PO)

    comptime _4er_kwargs = fn(PO, PO, PO, PO, OwnedKwargsDict[PO]) raises -> PO
    comptime _4r_kwargs = fn(PO, PO, PO, PO, OwnedKwargsDict[PO]) -> PO
    comptime _4e_kwargs = fn(PO, PO, PO, PO, OwnedKwargsDict[PO]) raises
    comptime _4_kwargs = fn(PO, PO, PO, PO, OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4er, Self.self_type], f: Self._4er
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4r, Self.self_type], f: Self._4r
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4e, Self.self_type], f: Self._4e
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._4, Self.self_type], f: Self._4
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._4_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._4_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 5 arguments (typed self methods - 4 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _5er_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) raises -> PO
    comptime _5r_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) -> PO
    comptime _5e_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    ) raises
    comptime _5_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO
    )

    comptime _5er_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) raises -> PO
    comptime _5r_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) -> PO
    comptime _5e_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) raises
    comptime _5_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    )

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5er_self, Self.self_type],
        f: Self._5er_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5r_self, Self.self_type],
        f: Self._5r_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5e_self, Self.self_type],
        f: Self._5e_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5_self, Self.self_type],
        f: Self._5_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5er_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5r_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5e_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 5 arguments
    # ===-------------------------------------------------------------------===#

    comptime _5er = fn(PO, PO, PO, PO, PO) raises -> PO
    comptime _5r = fn(PO, PO, PO, PO, PO) -> PO
    comptime _5e = fn(PO, PO, PO, PO, PO) raises
    comptime _5 = fn(PO, PO, PO, PO, PO)

    comptime _5er_kwargs = fn(
        PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) raises -> PO
    comptime _5r_kwargs = fn(PO, PO, PO, PO, PO, OwnedKwargsDict[PO]) -> PO
    comptime _5e_kwargs = fn(PO, PO, PO, PO, PO, OwnedKwargsDict[PO]) raises
    comptime _5_kwargs = fn(PO, PO, PO, PO, PO, OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5er, Self.self_type], f: Self._5er
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5r, Self.self_type], f: Self._5r
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5e, Self.self_type], f: Self._5e
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._5, Self.self_type], f: Self._5
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._5_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._5_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 6 arguments (typed self methods - 5 additional arguments)
    # ===-------------------------------------------------------------------===#

    comptime _6er_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) raises -> PO
    comptime _6r_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) -> PO
    comptime _6e_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    ) raises
    comptime _6_self = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin], PO, PO, PO, PO, PO
    )

    comptime _6er_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) raises -> PO
    comptime _6r_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) -> PO
    comptime _6e_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    ) raises
    comptime _6_self_kwargs = fn(
        UnsafePointer[Self.self_type, MutAnyOrigin],
        PO,
        PO,
        PO,
        PO,
        PO,
        OwnedKwargsDict[PO],
    )

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6er_self, Self.self_type],
        f: Self._6er_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6r_self, Self.self_type],
        f: Self._6r_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6e_self, Self.self_type],
        f: Self._6e_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6_self, Self.self_type],
        f: Self._6_self,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6er_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6er_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6r_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6r_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6e_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6e_self_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6_self_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6_self_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # 6 arguments
    # ===-------------------------------------------------------------------===#

    comptime _6er = fn(PO, PO, PO, PO, PO, PO) raises -> PO
    comptime _6r = fn(PO, PO, PO, PO, PO, PO) -> PO
    comptime _6e = fn(PO, PO, PO, PO, PO, PO) raises
    comptime _6 = fn(PO, PO, PO, PO, PO, PO)

    comptime _6er_kwargs = fn(
        PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]
    ) raises -> PO
    comptime _6r_kwargs = fn(PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]) -> PO
    comptime _6e_kwargs = fn(PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO]) raises
    comptime _6_kwargs = fn(PO, PO, PO, PO, PO, PO, OwnedKwargsDict[PO])

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6er, Self.self_type], f: Self._6er
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6r, Self.self_type], f: Self._6r
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6e, Self.self_type], f: Self._6e
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[Self._6, Self.self_type], f: Self._6
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6er_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6er_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6r_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6r_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6e_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6e_kwargs,
    ):
        self._func = f

    @doc_private
    @implicit
    fn __init__(
        out self: PyObjectFunction[
            Self._6_kwargs, Self.self_type, has_kwargs=True
        ],
        f: Self._6_kwargs,
    ):
        self._func = f

    # ===-------------------------------------------------------------------===#
    # Helper utilities
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @always_inline("nodebug")
    fn _get_self_arg(
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

        comptime if _type_is_eq[Self.self_type, NoneType]():
            comptime assert False, "Cannot get self arg for NoneType"
            # This line should never be reached due to the constraint
            abort("Unreachable code")
        else:
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
    fn _convert_kwargs(
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
    fn _has_type[other_func_type: __TypeOfAllTypes]() -> Bool:
        return _type_is_eq[Self.func_type, other_func_type]()

    @staticmethod
    @always_inline("nodebug")
    fn _has_arity(arity: Int) -> Bool:
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
        else:
            return False

    # ===-------------------------------------------------------------------===#
    # Call wrappers
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn _call_func(self, py_args: PO) raises -> PO:
        comptime if Self._has_arity(0):
            check_arguments_arity(0, py_args)

            comptime if self._has_type[Self._0er]():
                return rebind[Self._0er](self._func)()
            elif self._has_type[Self._0r]():
                return rebind[Self._0r](self._func)()
            elif self._has_type[Self._0e]():
                return PO(rebind[Self._0e](self._func)())
            elif self._has_type[Self._0]():
                return PO(rebind[Self._0](self._func)())
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(1):
            check_arguments_arity(1, py_args)
            var arg0 = py_args[0]

            comptime if self._has_type[Self._1er]():
                return rebind[Self._1er](self._func)(arg0)
            elif self._has_type[Self._1r]():
                return rebind[Self._1r](self._func)(arg0)
            elif self._has_type[Self._1e]():
                return PO(rebind[Self._1e](self._func)(arg0))
            elif self._has_type[Self._1]():
                return PO(rebind[Self._1](self._func)(arg0))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(2):
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]

            comptime if self._has_type[Self._2er]():
                return rebind[Self._2er](self._func)(arg0, arg1)
            elif self._has_type[Self._2r]():
                return rebind[Self._2r](self._func)(arg0, arg1)
            elif self._has_type[Self._2e]():
                return PO(rebind[Self._2e](self._func)(arg0, arg1))
            elif self._has_type[Self._2]():
                return PO(rebind[Self._2](self._func)(arg0, arg1))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(3):
            check_arguments_arity(3, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]

            comptime if self._has_type[Self._3er]():
                return rebind[Self._3er](self._func)(arg0, arg1, arg2)
            elif self._has_type[Self._3r]():
                return rebind[Self._3r](self._func)(arg0, arg1, arg2)
            elif self._has_type[Self._3e]():
                return PO(rebind[Self._3e](self._func)(arg0, arg1, arg2))
            elif self._has_type[Self._3]():
                return PO(rebind[Self._3](self._func)(arg0, arg1, arg2))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(4):
            check_arguments_arity(4, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]

            comptime if self._has_type[Self._4er]():
                return rebind[Self._4er](self._func)(arg0, arg1, arg2, arg3)
            elif self._has_type[Self._4r]():
                return rebind[Self._4r](self._func)(arg0, arg1, arg2, arg3)
            elif self._has_type[Self._4e]():
                return PO(rebind[Self._4e](self._func)(arg0, arg1, arg2, arg3))
            elif self._has_type[Self._4]():
                return PO(rebind[Self._4](self._func)(arg0, arg1, arg2, arg3))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(5):
            check_arguments_arity(5, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]
            var arg4 = py_args[4]

            comptime if self._has_type[Self._5er]():
                return rebind[Self._5er](self._func)(
                    arg0, arg1, arg2, arg3, arg4
                )
            elif self._has_type[Self._5r]():
                return rebind[Self._5r](self._func)(
                    arg0, arg1, arg2, arg3, arg4
                )
            elif self._has_type[Self._5e]():
                return PO(
                    rebind[Self._5e](self._func)(arg0, arg1, arg2, arg3, arg4)
                )
            elif self._has_type[Self._5]():
                return PO(
                    rebind[Self._5](self._func)(arg0, arg1, arg2, arg3, arg4)
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(6):
            check_arguments_arity(6, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]
            var arg4 = py_args[4]
            var arg5 = py_args[5]

            comptime if self._has_type[Self._6er]():
                return rebind[Self._6er](self._func)(
                    arg0, arg1, arg2, arg3, arg4, arg5
                )
            elif self._has_type[Self._6r]():
                return rebind[Self._6r](self._func)(
                    arg0, arg1, arg2, arg3, arg4, arg5
                )
            elif self._has_type[Self._6e]():
                return PO(
                    rebind[Self._6e](self._func)(
                        arg0, arg1, arg2, arg3, arg4, arg5
                    )
                )
            elif self._has_type[Self._6]():
                return PO(
                    rebind[Self._6](self._func)(
                        arg0, arg1, arg2, arg3, arg4, arg5
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        else:
            comptime assert False, "unsupported arity"
            return PO()

    @always_inline("nodebug")
    fn _call_func(self, py_args: PO, py_kwargs: PO) raises -> PO:
        comptime assert (
            Self.has_kwargs
        ), "should only be used for functions that accept kwargs"
        var kwargs = Self._convert_kwargs(py_kwargs)

        comptime if Self._has_arity(0):
            check_arguments_arity(0, py_args)

            comptime if self._has_type[Self._0er_kwargs]():
                return rebind[Self._0er_kwargs](self._func)(kwargs)
            elif self._has_type[Self._0r_kwargs]():
                return rebind[Self._0r_kwargs](self._func)(kwargs)
            elif self._has_type[Self._0e_kwargs]():
                return PO(rebind[Self._0e_kwargs](self._func)(kwargs))
            elif self._has_type[Self._0_kwargs]():
                return PO(rebind[Self._0_kwargs](self._func)(kwargs))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(1):
            check_arguments_arity(1, py_args)
            var arg0 = py_args[0]

            comptime if self._has_type[Self._1er_kwargs]():
                return rebind[Self._1er_kwargs](self._func)(arg0, kwargs)
            elif self._has_type[Self._1r_kwargs]():
                return rebind[Self._1r_kwargs](self._func)(arg0, kwargs)
            elif self._has_type[Self._1e_kwargs]():
                return PO(rebind[Self._1e_kwargs](self._func)(arg0, kwargs))
            elif self._has_type[Self._1_kwargs]():
                return PO(rebind[Self._1_kwargs](self._func)(arg0, kwargs))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(2):
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]

            comptime if self._has_type[Self._2er_kwargs]():
                return rebind[Self._2er_kwargs](self._func)(arg0, arg1, kwargs)
            elif self._has_type[Self._2r_kwargs]():
                return rebind[Self._2r_kwargs](self._func)(arg0, arg1, kwargs)
            elif self._has_type[Self._2e_kwargs]():
                return PO(
                    rebind[Self._2e_kwargs](self._func)(arg0, arg1, kwargs)
                )
            elif self._has_type[Self._2_kwargs]():
                return PO(
                    rebind[Self._2_kwargs](self._func)(arg0, arg1, kwargs)
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(3):
            check_arguments_arity(3, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]

            comptime if self._has_type[Self._3er_kwargs]():
                return rebind[Self._3er_kwargs](self._func)(
                    arg0, arg1, arg2, kwargs
                )
            elif self._has_type[Self._3r_kwargs]():
                return rebind[Self._3r_kwargs](self._func)(
                    arg0, arg1, arg2, kwargs
                )
            elif self._has_type[Self._3e_kwargs]():
                return PO(
                    rebind[Self._3e_kwargs](self._func)(
                        arg0, arg1, arg2, kwargs
                    )
                )
            elif self._has_type[Self._3_kwargs]():
                return PO(
                    rebind[Self._3_kwargs](self._func)(arg0, arg1, arg2, kwargs)
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(4):
            check_arguments_arity(4, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]

            comptime if self._has_type[Self._4er_kwargs]():
                return rebind[Self._4er_kwargs](self._func)(
                    arg0, arg1, arg2, arg3, kwargs
                )
            elif self._has_type[Self._4r_kwargs]():
                return rebind[Self._4r_kwargs](self._func)(
                    arg0, arg1, arg2, arg3, kwargs
                )
            elif self._has_type[Self._4e_kwargs]():
                return PO(
                    rebind[Self._4e_kwargs](self._func)(
                        arg0, arg1, arg2, arg3, kwargs
                    )
                )
            elif self._has_type[Self._4_kwargs]():
                return PO(
                    rebind[Self._4_kwargs](self._func)(
                        arg0, arg1, arg2, arg3, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(5):
            check_arguments_arity(5, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]
            var arg4 = py_args[4]

            comptime if self._has_type[Self._5er_kwargs]():
                return rebind[Self._5er_kwargs](self._func)(
                    arg0, arg1, arg2, arg3, arg4, kwargs
                )
            elif self._has_type[Self._5r_kwargs]():
                return rebind[Self._5r_kwargs](self._func)(
                    arg0, arg1, arg2, arg3, arg4, kwargs
                )
            elif self._has_type[Self._5e_kwargs]():
                return PO(
                    rebind[Self._5e_kwargs](self._func)(
                        arg0, arg1, arg2, arg3, arg4, kwargs
                    )
                )
            elif self._has_type[Self._5_kwargs]():
                return PO(
                    rebind[Self._5_kwargs](self._func)(
                        arg0, arg1, arg2, arg3, arg4, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(6):
            check_arguments_arity(6, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]
            var arg4 = py_args[4]
            var arg5 = py_args[5]

            comptime if self._has_type[Self._6er_kwargs]():
                return rebind[Self._6er_kwargs](self._func)(
                    arg0, arg1, arg2, arg3, arg4, arg5, kwargs
                )
            elif self._has_type[Self._6r_kwargs]():
                return rebind[Self._6r_kwargs](self._func)(
                    arg0, arg1, arg2, arg3, arg4, arg5, kwargs
                )
            elif self._has_type[Self._6e_kwargs]():
                return PO(
                    rebind[Self._6e_kwargs](self._func)(
                        arg0, arg1, arg2, arg3, arg4, arg5, kwargs
                    )
                )
            elif self._has_type[Self._6_kwargs]():
                return PO(
                    rebind[Self._6_kwargs](self._func)(
                        arg0, arg1, arg2, arg3, arg4, arg5, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        else:
            comptime assert False, "unsupported arity"
            return PO()

    @always_inline("nodebug")
    fn _call_method(self, py_self: PO, py_args: PO) raises -> PO:
        comptime assert not Self._has_arity(0), "method arity must not be 0"

        comptime if Self._has_arity(1):
            check_arguments_arity(0, py_args)

            comptime if self._has_type[Self._1er]():
                return rebind[Self._1er](self._func)(py_self)
            elif self._has_type[Self._1r]():
                return rebind[Self._1r](self._func)(py_self)
            elif self._has_type[Self._1e]():
                return PO(rebind[Self._1e](self._func)(py_self))
            elif self._has_type[Self._1]():
                return PO(rebind[Self._1](self._func)(py_self))
            elif self._has_type[Self._1er_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._1er_self](self._func)(self_arg)
            elif self._has_type[Self._1r_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._1r_self](self._func)(self_arg)
            elif self._has_type[Self._1e_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(rebind[Self._1e_self](self._func)(self_arg))
            elif self._has_type[Self._1_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(rebind[Self._1_self](self._func)(self_arg))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(2):
            check_arguments_arity(1, py_args)
            var arg0 = py_args[0]

            comptime if self._has_type[Self._2er]():
                return rebind[Self._2er](self._func)(py_self, arg0)
            elif self._has_type[Self._2r]():
                return rebind[Self._2r](self._func)(py_self, arg0)
            elif self._has_type[Self._2e]():
                return PO(rebind[Self._2e](self._func)(py_self, arg0))
            elif self._has_type[Self._2]():
                return PO(rebind[Self._2](self._func)(py_self, arg0))
            elif self._has_type[Self._2er_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._2er_self](self._func)(self_arg, arg0)
            elif self._has_type[Self._2r_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._2r_self](self._func)(self_arg, arg0)
            elif self._has_type[Self._2e_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(rebind[Self._2e_self](self._func)(self_arg, arg0))
            elif self._has_type[Self._2_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(rebind[Self._2_self](self._func)(self_arg, arg0))
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(3):
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]

            comptime if self._has_type[Self._3er]():
                return rebind[Self._3er](self._func)(py_self, arg0, arg1)
            elif self._has_type[Self._3r]():
                return rebind[Self._3r](self._func)(py_self, arg0, arg1)
            elif self._has_type[Self._3e]():
                return PO(rebind[Self._3e](self._func)(py_self, arg0, arg1))
            elif self._has_type[Self._3]():
                return PO(rebind[Self._3](self._func)(py_self, arg0, arg1))
            elif self._has_type[Self._3er_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._3er_self](self._func)(self_arg, arg0, arg1)
            elif self._has_type[Self._3r_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._3r_self](self._func)(self_arg, arg0, arg1)
            elif self._has_type[Self._3e_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._3e_self](self._func)(self_arg, arg0, arg1)
                )
            elif self._has_type[Self._3_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._3_self](self._func)(self_arg, arg0, arg1)
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(4):
            check_arguments_arity(3, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]

            comptime if self._has_type[Self._4er]():
                return rebind[Self._4er](self._func)(py_self, arg0, arg1, arg2)
            elif self._has_type[Self._4r]():
                return rebind[Self._4r](self._func)(py_self, arg0, arg1, arg2)
            elif self._has_type[Self._4e]():
                return PO(
                    rebind[Self._4e](self._func)(py_self, arg0, arg1, arg2)
                )
            elif self._has_type[Self._4]():
                return PO(
                    rebind[Self._4](self._func)(py_self, arg0, arg1, arg2)
                )
            elif self._has_type[Self._4er_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._4er_self](self._func)(
                    self_arg, arg0, arg1, arg2
                )
            elif self._has_type[Self._4r_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._4r_self](self._func)(
                    self_arg, arg0, arg1, arg2
                )
            elif self._has_type[Self._4e_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._4e_self](self._func)(
                        self_arg, arg0, arg1, arg2
                    )
                )
            elif self._has_type[Self._4_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._4_self](self._func)(self_arg, arg0, arg1, arg2)
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(5):
            check_arguments_arity(4, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]

            comptime if self._has_type[Self._5er]():
                return rebind[Self._5er](self._func)(
                    py_self, arg0, arg1, arg2, arg3
                )
            elif self._has_type[Self._5r]():
                return rebind[Self._5r](self._func)(
                    py_self, arg0, arg1, arg2, arg3
                )
            elif self._has_type[Self._5e]():
                return PO(
                    rebind[Self._5e](self._func)(
                        py_self, arg0, arg1, arg2, arg3
                    )
                )
            elif self._has_type[Self._5]():
                return PO(
                    rebind[Self._5](self._func)(py_self, arg0, arg1, arg2, arg3)
                )
            elif self._has_type[Self._5er_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._5er_self](self._func)(
                    self_arg, arg0, arg1, arg2, arg3
                )
            elif self._has_type[Self._5r_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._5r_self](self._func)(
                    self_arg, arg0, arg1, arg2, arg3
                )
            elif self._has_type[Self._5e_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._5e_self](self._func)(
                        self_arg, arg0, arg1, arg2, arg3
                    )
                )
            elif self._has_type[Self._5_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._5_self](self._func)(
                        self_arg, arg0, arg1, arg2, arg3
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(6):
            check_arguments_arity(5, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]
            var arg4 = py_args[4]

            comptime if self._has_type[Self._6er]():
                return rebind[Self._6er](self._func)(
                    py_self, arg0, arg1, arg2, arg3, arg4
                )
            elif self._has_type[Self._6r]():
                return rebind[Self._6r](self._func)(
                    py_self, arg0, arg1, arg2, arg3, arg4
                )
            elif self._has_type[Self._6e]():
                return PO(
                    rebind[Self._6e](self._func)(
                        py_self, arg0, arg1, arg2, arg3, arg4
                    )
                )
            elif self._has_type[Self._6]():
                return PO(
                    rebind[Self._6](self._func)(
                        py_self, arg0, arg1, arg2, arg3, arg4
                    )
                )
            elif self._has_type[Self._6er_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._6er_self](self._func)(
                    self_arg, arg0, arg1, arg2, arg3, arg4
                )
            elif self._has_type[Self._6r_self]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._6r_self](self._func)(
                    self_arg, arg0, arg1, arg2, arg3, arg4
                )
            elif self._has_type[Self._6e_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._6e_self](self._func)(
                        self_arg, arg0, arg1, arg2, arg3, arg4
                    )
                )
            elif self._has_type[Self._6_self]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._6_self](self._func)(
                        self_arg, arg0, arg1, arg2, arg3, arg4
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        else:
            comptime assert False, "unsupported arity"
            return PO()

    @always_inline("nodebug")
    fn _call_method(self, py_self: PO, py_args: PO, py_kwargs: PO) raises -> PO:
        comptime assert not Self._has_arity(0), "method arity must not be 0"
        comptime assert (
            Self.has_kwargs
        ), "should only be used for methods that accept kwargs"
        var kwargs = Self._convert_kwargs(py_kwargs)

        comptime if Self._has_arity(1):
            check_arguments_arity(0, py_args)

            comptime if self._has_type[Self._1er_kwargs]():
                return rebind[Self._1er_kwargs](self._func)(py_self, kwargs)
            elif self._has_type[Self._1r_kwargs]():
                return rebind[Self._1r_kwargs](self._func)(py_self, kwargs)
            elif self._has_type[Self._1e_kwargs]():
                return PO(rebind[Self._1e_kwargs](self._func)(py_self, kwargs))
            elif self._has_type[Self._1_kwargs]():
                return PO(rebind[Self._1_kwargs](self._func)(py_self, kwargs))
            elif self._has_type[Self._1er_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._1er_self_kwargs](self._func)(
                    self_arg, kwargs
                )
            elif self._has_type[Self._1r_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._1r_self_kwargs](self._func)(
                    self_arg, kwargs
                )
            elif self._has_type[Self._1e_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._1e_self_kwargs](self._func)(self_arg, kwargs)
                )
            elif self._has_type[Self._1_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._1_self_kwargs](self._func)(self_arg, kwargs)
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(2):
            check_arguments_arity(1, py_args)
            var arg0 = py_args[0]

            comptime if self._has_type[Self._2er_kwargs]():
                return rebind[Self._2er_kwargs](self._func)(
                    py_self, arg0, kwargs
                )
            elif self._has_type[Self._2r_kwargs]():
                return rebind[Self._2r_kwargs](self._func)(
                    py_self, arg0, kwargs
                )
            elif self._has_type[Self._2e_kwargs]():
                return PO(
                    rebind[Self._2e_kwargs](self._func)(py_self, arg0, kwargs)
                )
            elif self._has_type[Self._2_kwargs]():
                return PO(
                    rebind[Self._2_kwargs](self._func)(py_self, arg0, kwargs)
                )
            elif self._has_type[Self._2er_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._2er_self_kwargs](self._func)(
                    self_arg, arg0, kwargs
                )
            elif self._has_type[Self._2r_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._2r_self_kwargs](self._func)(
                    self_arg, arg0, kwargs
                )
            elif self._has_type[Self._2e_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._2e_self_kwargs](self._func)(
                        self_arg, arg0, kwargs
                    )
                )
            elif self._has_type[Self._2_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._2_self_kwargs](self._func)(
                        self_arg, arg0, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(3):
            check_arguments_arity(2, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]

            comptime if self._has_type[Self._3er_kwargs]():
                return rebind[Self._3er_kwargs](self._func)(
                    py_self, arg0, arg1, kwargs
                )
            elif self._has_type[Self._3r_kwargs]():
                return rebind[Self._3r_kwargs](self._func)(
                    py_self, arg0, arg1, kwargs
                )
            elif self._has_type[Self._3e_kwargs]():
                return PO(
                    rebind[Self._3e_kwargs](self._func)(
                        py_self, arg0, arg1, kwargs
                    )
                )
            elif self._has_type[Self._3_kwargs]():
                return PO(
                    rebind[Self._3_kwargs](self._func)(
                        py_self, arg0, arg1, kwargs
                    )
                )
            elif self._has_type[Self._3er_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._3er_self_kwargs](self._func)(
                    self_arg, arg0, arg1, kwargs
                )
            elif self._has_type[Self._3r_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._3r_self_kwargs](self._func)(
                    self_arg, arg0, arg1, kwargs
                )
            elif self._has_type[Self._3e_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._3e_self_kwargs](self._func)(
                        self_arg, arg0, arg1, kwargs
                    )
                )
            elif self._has_type[Self._3_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._3_self_kwargs](self._func)(
                        self_arg, arg0, arg1, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(4):
            check_arguments_arity(3, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]

            comptime if self._has_type[Self._4er_kwargs]():
                return rebind[Self._4er_kwargs](self._func)(
                    py_self, arg0, arg1, arg2, kwargs
                )
            elif self._has_type[Self._4r_kwargs]():
                return rebind[Self._4r_kwargs](self._func)(
                    py_self, arg0, arg1, arg2, kwargs
                )
            elif self._has_type[Self._4e_kwargs]():
                return PO(
                    rebind[Self._4e_kwargs](self._func)(
                        py_self, arg0, arg1, arg2, kwargs
                    )
                )
            elif self._has_type[Self._4_kwargs]():
                return PO(
                    rebind[Self._4_kwargs](self._func)(
                        py_self, arg0, arg1, arg2, kwargs
                    )
                )
            elif self._has_type[Self._4er_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._4er_self_kwargs](self._func)(
                    self_arg, arg0, arg1, arg2, kwargs
                )
            elif self._has_type[Self._4r_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._4r_self_kwargs](self._func)(
                    self_arg, arg0, arg1, arg2, kwargs
                )
            elif self._has_type[Self._4e_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._4e_self_kwargs](self._func)(
                        self_arg, arg0, arg1, arg2, kwargs
                    )
                )
            elif self._has_type[Self._4_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._4_self_kwargs](self._func)(
                        self_arg, arg0, arg1, arg2, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(5):
            check_arguments_arity(4, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]

            comptime if self._has_type[Self._5er_kwargs]():
                return rebind[Self._5er_kwargs](self._func)(
                    py_self, arg0, arg1, arg2, arg3, kwargs
                )
            elif self._has_type[Self._5r_kwargs]():
                return rebind[Self._5r_kwargs](self._func)(
                    py_self, arg0, arg1, arg2, arg3, kwargs
                )
            elif self._has_type[Self._5e_kwargs]():
                return PO(
                    rebind[Self._5e_kwargs](self._func)(
                        py_self, arg0, arg1, arg2, arg3, kwargs
                    )
                )
            elif self._has_type[Self._5_kwargs]():
                return PO(
                    rebind[Self._5_kwargs](self._func)(
                        py_self, arg0, arg1, arg2, arg3, kwargs
                    )
                )
            elif self._has_type[Self._5er_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._5er_self_kwargs](self._func)(
                    self_arg, arg0, arg1, arg2, arg3, kwargs
                )
            elif self._has_type[Self._5r_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._5r_self_kwargs](self._func)(
                    self_arg, arg0, arg1, arg2, arg3, kwargs
                )
            elif self._has_type[Self._5e_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._5e_self_kwargs](self._func)(
                        self_arg, arg0, arg1, arg2, arg3, kwargs
                    )
                )
            elif self._has_type[Self._5_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._5_self_kwargs](self._func)(
                        self_arg, arg0, arg1, arg2, arg3, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        elif Self._has_arity(6):
            check_arguments_arity(5, py_args)
            var arg0 = py_args[0]
            var arg1 = py_args[1]
            var arg2 = py_args[2]
            var arg3 = py_args[3]
            var arg4 = py_args[4]

            comptime if self._has_type[Self._6er_kwargs]():
                return rebind[Self._6er_kwargs](self._func)(
                    py_self, arg0, arg1, arg2, arg3, arg4, kwargs
                )
            elif self._has_type[Self._6r_kwargs]():
                return rebind[Self._6r_kwargs](self._func)(
                    py_self, arg0, arg1, arg2, arg3, arg4, kwargs
                )
            elif self._has_type[Self._6e_kwargs]():
                return PO(
                    rebind[Self._6e_kwargs](self._func)(
                        py_self, arg0, arg1, arg2, arg3, arg4, kwargs
                    )
                )
            elif self._has_type[Self._6_kwargs]():
                return PO(
                    rebind[Self._6_kwargs](self._func)(
                        py_self, arg0, arg1, arg2, arg3, arg4, kwargs
                    )
                )
            elif self._has_type[Self._6er_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._6er_self_kwargs](self._func)(
                    self_arg, arg0, arg1, arg2, arg3, arg4, kwargs
                )
            elif self._has_type[Self._6r_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return rebind[Self._6r_self_kwargs](self._func)(
                    self_arg, arg0, arg1, arg2, arg3, arg4, kwargs
                )
            elif self._has_type[Self._6e_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._6e_self_kwargs](self._func)(
                        self_arg, arg0, arg1, arg2, arg3, arg4, kwargs
                    )
                )
            elif self._has_type[Self._6_self_kwargs]():
                var self_arg = Self._get_self_arg(py_self)
                return PO(
                    rebind[Self._6_self_kwargs](self._func)(
                        self_arg, arg0, arg1, arg2, arg3, arg4, kwargs
                    )
                )
            else:
                comptime assert False, "unsupported signature"
                return PO()
        else:
            comptime assert False, "unsupported arity"
            return PO()
