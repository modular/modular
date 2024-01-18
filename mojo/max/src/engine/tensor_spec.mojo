# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import *
from ._status import *
from tensor import TensorSpec
from ._dtypes import EngineDType
from collections.vector import DynamicVector
from .session import InferenceSession
from ._tensor_spec_impl import *


@value
struct EngineTensorSpec:
    var ptr: CTensorSpec
    var lib: DLHandle
    var session: InferenceSession

    alias NewTensorSpecFnName = "M_newTensorSpec"

    fn __init__(
        inout self,
        ptr: CTensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __init__(
        inout self,
        name: StringRef,
        spec: TensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        let dtype: EngineDType = spec.dtype()
        let rank = spec.rank()
        var shape = DynamicVector[Int64]()
        let name_str: CString = name.data
        for i in range(rank):
            shape.push_back(spec[i])
        self.ptr = call_dylib_func[CTensorSpec](
            lib, Self.NewTensorSpecFnName, shape.data, rank, dtype, name_str
        )
        _ = name
        _ = shape
        self.lib = lib
        self.session = session ^

    fn __getitem__(self, idx: Int) -> Int:
        return self.ptr.get_dim_at(idx, self.lib)

    fn get_as_tensor_spec(self) -> TensorSpec:
        if self.ptr.is_dynamically_ranked(self.lib):
            print("tensors with dynamic rank are not supported here")
            trap()
        var shape = DynamicVector[Int]()
        let rank = self.ptr.get_rank(self.lib)
        for i in range(rank):
            shape.push_back(self[i])
        let dtype = self.ptr.get_dtype(self.lib)
        let spec = TensorSpec(dtype.to_dtype(), shape)
        return spec

    fn get_name(self) -> String:
        return self.ptr.get_name(self.lib)

    fn __str__(self) -> String:
        var _repr: String = "{name="
        _repr += self.get_name()
        _repr += ", spec="
        _repr += self.get_as_tensor_spec().__str__()
        _repr += "}"
        return _repr

    fn borrow_ptr(self) -> CTensorSpec:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
