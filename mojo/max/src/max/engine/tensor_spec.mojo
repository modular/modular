# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Defines `EngineTensorSpec` to define a model's input and output specs.
"""

from collections import List
from collections.optional import Optional
from os import abort
from sys.ffi import DLHandle

from max._utils import call_dylib_func
from max.tensor import TensorSpec

from ._tensor_spec_impl import CTensorSpec
from .session import InferenceSession


struct EngineTensorSpec(Stringable, CollectionElement):
    """
    Describes the input and output tensor specifications of a
    [`Model`](/max/api/mojo/engine/model/Model).

    Holds a description of the input/output tensor, given to or produced by
    MAX Engine. This is similar to
    [`TensorSpec`](/mojo/stdlib/tensor/tensor_spec/TensorSpec) in the Mojo
    standard library, but is specific to MAX Engine.

    This is the data type returned from
    [`Model.get_model_input_metadata()`](/max/api/mojo/engine/model/Model#get_model_input_metadata)
    and
    [`Model.get_model_output_metadata()`](/max/api/mojo/engine/model/Model#get_model_output_metadata).
    """

    var _ptr: CTensorSpec
    var _lib: DLHandle
    var _session: InferenceSession

    alias _NewTensorSpecFnName = "M_newTensorSpec"

    fn __copyinit__(out self, other: Self):
        """Copy constructor for Tensor Spec.

        Args:
            other: Instance of TensorSpec to be copied.
        """
        self = Self(
            other.get_name(),
            other.get_shape(),
            other.get_dtype(),
            other._lib,
            other._session,
        )

    @always_inline
    fn copy(self) -> Self:
        """Explicitly construct a copy of self.

        Returns:
            A copy of this value.
        """
        return self

    fn __moveinit__(out self, owned existing: Self):
        """Move constructor for Tensor Spec.

        Args:
            existing: Instance of TensorSpec to be moved.
        """
        self._ptr = existing._ptr
        self._lib = existing._lib
        self._session = existing._session^

    fn __init__(
        out self,
        ptr: CTensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Construct EngineTensorSpec.
        Do not use this function directly.
        Use functions from InferenceSession to create EngineTensorSpec.

        Args:
            ptr: C API pointer of TensorSpec.
            lib: Handle to library.
            session: Copy of InferenceSession from which this instance
                     was created.
        """
        self._ptr = ptr
        self._lib = lib
        self._session = session^

    fn __init__(
        out self,
        name: String,
        spec: TensorSpec,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Creates an instance of EngineTensorSpec.
        Do not use this function directly.
        Use functions from InferenceSession to create EngineTensorSpec.

        Args:
            name: Name of tensor.
            spec: Descritpion of Tensor in stdlib `TensorSpec` format.
            lib: Handle to the library.
            session: Copy of InferenceSession from which this instance
                     was created.
        """
        var dtype = spec.dtype()
        var rank = spec.rank()
        var shape = List[Int64]()
        var name_str = name.unsafe_ptr()
        for i in range(rank):
            shape.append(spec[i])
        self._ptr = call_dylib_func[CTensorSpec](
            lib,
            Self._NewTensorSpecFnName,
            shape.data,
            rank,
            dtype,
            name_str,
        )
        _ = name
        _ = shape
        self._lib = lib
        self._session = session^

    fn __init__(
        out self,
        name: String,
        shape: Optional[List[Optional[Int]]],
        dtype: DType,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Creates an instance of EngineTensorSpec.
        Do not use this function directly.
        Use functions from InferenceSession to create EngineTensorSpec.
        Note that this almost similar to another constructor.
        This accepts Int instead of Int64 shape types.

        Args:
            name: Name of tensor.
            shape: Shape of the tensor.
            dtype: DataType of the tensor.
            lib: Handle to the library.
            session: Copy of InferenceSession from which this instance
                     was created.
        """

        if not shape:
            self = Self(
                name,
                Optional[List[Optional[Int64]]](None),
                dtype,
                lib,
                session,
            )
        else:
            var casted_shape = List[Optional[Int64]]()
            for dim in shape.value():
                if not dim[]:
                    casted_shape.append(None)
                else:
                    casted_shape.append(Int64(dim[].value()))
            self = Self(name, casted_shape, dtype, lib, session)

    fn __init__(
        out self,
        name: String,
        shape: Optional[List[Optional[Int64]]],
        dtype: DType,
        lib: DLHandle,
        owned session: InferenceSession,
    ):
        """Creates an instance of EngineTensorSpec.
        Do not use this function directly.
        Use functions from InferenceSession to create EngineTensorSpec.
        Note that this almost similar to another constructor.
        This accepts Int64 instead of Int shape types.

        Args:
            name: Name of tensor.
            shape: Shape of the tensor.
            dtype: DataType of the tensor.
            lib: Handle to the library.
            session: Copy of InferenceSession from which this instance
                     was created.
        """
        var name_str = name.unsafe_ptr()
        if shape:
            var inner_shape = shape.value()
            var rank = len(inner_shape)
            var adjusted_shape = List[Int64]()
            adjusted_shape.reserve(rank)
            var dynamic_value = CTensorSpec.get_dynamic_dimension_value(lib)
            for i in range(rank):
                var dim = inner_shape[i]
                if not dim:
                    adjusted_shape.append(dynamic_value)
                else:
                    adjusted_shape.append(dim.value())
            self._ptr = call_dylib_func[CTensorSpec](
                lib,
                Self._NewTensorSpecFnName,
                adjusted_shape.data,
                rank,
                dtype,
                name_str,
            )
            _ = adjusted_shape^
        else:
            self._ptr = call_dylib_func[CTensorSpec](
                lib,
                Self._NewTensorSpecFnName,
                CTensorSpec.ptr_type(),
                CTensorSpec.get_dynamic_rank_value(lib),
                dtype,
                name_str,
            )
        _ = name
        _ = shape
        self._lib = lib
        self._session = session^

    fn __getitem__(self, idx: Int) raises -> Optional[Int]:
        """Get the dimension at the given index.

        Args:
            idx: Index to get the dimension.

        Returns:
            Dimension as integer if dimension is static, else None.

        Raises:
            Raise error if spec has no static rank.
        """

        if self._ptr.is_dynamically_ranked(self._lib):
            raise "spec is dynamically ranked"

        var dim = self._ptr.get_dim_at(idx, self._lib)
        if dim == CTensorSpec.get_dynamic_dimension_value(self._lib):
            return None
        return dim

    fn rank(self) -> Optional[Int]:
        """Gets the rank of spec.

        Returns:
            Rank if rank is static, else None.
        """
        if not self.has_rank():
            return None
        return self._ptr.get_rank(self._lib)

    fn has_rank(self) -> Bool:
        """Check if the spec has static rank.

        Returns:
            True if spec has static rank, else False.
        """
        return not self._ptr.is_dynamically_ranked(self._lib)

    fn get_as_tensor_spec(self) raises -> TensorSpec:
        """Get the Mojo TensorSpec equivalent of Engine TensorSpec.

        Returns:
            Spec in Mojo TensorSpec format.

        Raises
            Raise error if spec has dynamic rank.
        """
        var rank_or = self.rank()
        if not rank_or:
            raise "tensors with dynamic rank cannot be converted to Mojo TensorSpec."

        var shape = List[Int, hint_trivial_type=True]()
        var rank = rank_or.value()
        for i in range(rank):
            shape.append(self[i].value())
        var dtype = self._ptr.get_dtype(self._lib)
        var spec = TensorSpec(dtype, shape)
        return spec

    fn get_name(self) -> String:
        """Gets the name of tensor corresponding to spec.

        Returns:
            Name of the Tensor as String.
        """
        return self._ptr.get_name(self._lib)

    fn get_dtype(self) -> DType:
        """Gets the DType of tensor corresponding to spec.

        Returns:
            DType of the Tensor.
        """
        return self._ptr.get_dtype(self._lib)

    fn get_shape(self) -> Optional[List[Optional[Int]]]:
        """Gets the shape of tensor corresponding to spec.

        Returns:
            Shape of the Tensor. Returns None if rank is dynamic.
        """
        var rank_or = self.rank()
        if not rank_or:
            return None

        var shape_list = List[Optional[Int]]()
        var rank = rank_or.value()
        for i in range(rank):
            var dim: Optional[Int]
            try:
                dim = self[i]
            except err:
                abort("unreachable condition")
                while True:
                    pass

            if not dim:
                shape_list.append(None)
            else:
                shape_list.append(dim.value())

        return shape_list

    fn __str__(self) -> String:
        """Gets the String representation of Spec.

        Returns:
            Spec as string. This will be of format `{name=<spec name>, spec=[None|shape]xdtype}`.
        """
        var _repr: String = "{name="
        _repr += self.get_name()
        _repr += ", spec="

        var shape_list = self.get_shape()

        if not shape_list:
            _repr += "None x "
        else:
            for dim in shape_list.value():
                if not dim[]:
                    _repr += "-1"
                else:
                    _repr += String(dim[].value())
                _repr += "x"
        _repr += String(self.get_dtype())
        _repr += "}"
        return _repr

    fn _borrow_ptr(self) -> CTensorSpec:
        return self._ptr

    fn __del__(owned self):
        """Destructor for EngineTensorSpec."""
        self._ptr.free(self._lib)
        _ = self._session^
