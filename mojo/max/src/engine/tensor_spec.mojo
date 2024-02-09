# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.unsafe import DTypePointer
from sys.ffi import DLHandle
from ._utils import call_dylib_func
from tensor import TensorSpec
from ._dtypes import EngineDType
from collections.vector import DynamicVector
from collections.optional import Optional
from .session import InferenceSession
from ._tensor_spec_impl import CTensorSpec


struct EngineTensorSpec(Stringable, Movable):
    var ptr: CTensorSpec
    var lib: DLHandle
    var session: InferenceSession

    alias NewTensorSpecFnName = "M_newTensorSpec"

    fn __moveinit__(inout self, owned existing: Self):
        self.ptr = existing.ptr
        self.lib = existing.lib
        self.session = existing.session ^

    fn __init__(
        inout self,
        ptr: CTensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Construct EngineTensorSpec.
        Do not use this function directly.
        Use functions from InferenceSession to create EngineTensorSpec.
        """
        self.ptr = ptr
        self.lib = lib
        self.session = session ^

    fn __init__(
        inout self,
        name: String,
        spec: TensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        let dtype = spec.dtype()
        let rank = spec.rank()
        var shape = DynamicVector[Int64]()
        let name_str = name._as_ptr()
        for i in range(rank):
            shape.push_back(spec[i])
        self.ptr = call_dylib_func[CTensorSpec](
            lib,
            Self.NewTensorSpecFnName,
            shape.data,
            rank,
            EngineDType(dtype),
            name_str,
        )
        _ = name
        _ = shape
        self.lib = lib
        self.session = session ^

    fn __init__(
        inout self,
        name: String,
        shape: Optional[DynamicVector[Optional[Int64]]],
        dtype: DType,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        let name_str = name._as_ptr()
        if shape:
            let inner_shape = shape.value()
            let rank = len(inner_shape)
            var adjusted_shape = DynamicVector[Int64]()
            adjusted_shape.reserve(rank)
            let dynamic_value = CTensorSpec.get_dynamic_dimension_value(lib)
            for i in range(rank):
                let dim = inner_shape[i]
                if not dim:
                    adjusted_shape.push_back(dynamic_value)
                else:
                    adjusted_shape.push_back(dim.value())
            self.ptr = call_dylib_func[CTensorSpec](
                lib,
                Self.NewTensorSpecFnName,
                adjusted_shape.data,
                rank,
                EngineDType(dtype),
                name_str,
            )
            _ = adjusted_shape ^
        else:
            self.ptr = call_dylib_func[CTensorSpec](
                lib,
                Self.NewTensorSpecFnName,
                CTensorSpec.ptr_type(),
                CTensorSpec.get_dynamic_rank_value(lib),
                EngineDType(dtype),
                name_str,
            )
        _ = name
        _ = shape
        self.lib = lib
        self.session = session ^

    fn __getitem__(self, idx: Int) raises -> Optional[Int]:
        """Get the dimension at the given index.

        Args:
            idx: Index to get the dimension.

        Returns:
            Dimension as integer if dimension is static, else None.

        Raises:
            Raise error if spec has no static rank.
        """

        if self.ptr.is_dynamically_ranked(self.lib):
            raise "spec is dynamically ranked"

        let dim = self.ptr.get_dim_at(idx, self.lib)
        if dim == CTensorSpec.get_dynamic_dimension_value(self.lib):
            return None
        return dim

    fn rank(self) -> Optional[Int]:
        """Gets the rank of spec.

        Returns:
            Rank if rank is static, else None.
        """
        if not self.has_rank():
            return None
        return self.ptr.get_rank(self.lib)

    fn has_rank(self) -> Bool:
        """Check if the spec has static rank.

        Returns:
            True if spec has static rank, else False.
        """
        return not self.ptr.is_dynamically_ranked(self.lib)

    fn get_as_tensor_spec(self) raises -> TensorSpec:
        """Get the Mojo TensorSpec equivalent of Engine TensorSpec.

        Returns
            Spec in Mojo TensorSpec format.

        Raises
            Raise error if spec has dynamic rank.
        """
        let rank_or = self.rank()
        if not rank_or:
            raise "tensors with dynamic rank cannot be converted to Mojo TensorSpec."

        var shape = DynamicVector[Int]()
        let rank = rank_or.value()
        for i in range(rank):
            shape.push_back(self[i].value())
        let dtype = self.ptr.get_dtype(self.lib)
        let spec = TensorSpec(dtype.to_dtype(), shape)
        return spec

    fn get_name(self) -> String:
        """Gets the name of tensor corresponding to spec.

        Returns:
            Name of the Tensor as String.
        """
        return self.ptr.get_name(self.lib)

    fn __str__(self) -> String:
        """Gets the String representation of Spec.

        Returns:
            Spec as string. This will be of format `{name=<spec name>, spec=[None|shape]xdtype}`.
        """
        var _repr: String = "{name="
        _repr += self.get_name()
        _repr += ", spec="
        let rank_or = self.rank()
        if not rank_or:
            _repr += "None x "
            _repr += str(self.ptr.get_dtype(self.lib).to_dtype())
        else:
            let rank = rank_or.value()
            for i in range(rank):
                let dim: Optional[Int]
                try:
                    dim = self[i]
                except err:
                    trap("unreachable condition")

                if not dim:
                    _repr += "-1x"
                else:
                    _repr += str(dim.value()) + "x"
            _repr += str(self.ptr.get_dtype(self.lib).to_dtype())
        _repr += "}"
        return _repr

    fn _borrow_ptr(self) -> CTensorSpec:
        return self.ptr

    fn __del__(owned self):
        self.ptr.free(self.lib)
        _ = self.session ^
