# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import Tensor, TensorSpec, TensorShape

from .type import Arity
from .capi import AttrPtr, LocPtr, ModulePtr, TypePtr
from .symbol import Symbol
from .type import MOTensor, MOType


@value
struct Module:
    var m: ModulePtr

    # ===------------------------------------------------------------------=== #
    # Constructors and basic accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self):
        self.__init__(capi.module_new())

    fn __str__(self) -> String:
        return capi.module_to_string(self.m)

    # ===------------------------------------------------------------------=== #
    # High level utilities
    # ===------------------------------------------------------------------=== #

    fn verify(self) raises:
        if not capi.module_verify(self.m):
            raise "Module did not verify"

    fn save_to_file(self, path: Path) raises:
        if not capi.module_to_bytecode(self.m, path.__str__()):
            raise "Error writing to file"

    # ===------------------------------------------------------------------=== #
    # Location factories
    # ===------------------------------------------------------------------=== #

    fn unknown_loc(self) -> LocPtr:
        return capi.loc_new_unknown(self.m)

    # ===------------------------------------------------------------------=== #
    # Attribute factories
    # ===------------------------------------------------------------------=== #

    fn tensor_attr[
        dtype: DType
    ](self, name: StringRef, owned value: Tensor[dtype]) -> AttrPtr:
        let t = MOTensor(value.spec()).to_mlir(self)
        return capi.attr_new_tensor(
            self.m,
            name,
            value._steal_ptr().bitcast[DType.invalid](),
            t,
            is_owned=True,
        )

    fn tensor_resource_attr(
        self, name: StringRef, file_name: StringRef, type: MOTensor
    ) -> AttrPtr:
        return capi.attr_new_tensor_from_file(
            self.m, name, file_name, type.to_mlir(self)
        )

    fn vector_attr[
        dtype: DType
    ](self, name: StringRef, values: DynamicVector[Scalar[dtype]]) -> AttrPtr:
        return capi.attr_new_tensor(
            self.m,
            name,
            values,
            MOTensor(dtype, len(values)).to_mlir(self),
            is_owned=False,
        )

    fn scalar_attr[
        dtype: DType
    ](
        self, name: StringRef, value: Scalar[dtype], rank: Int = 0
    ) raises -> AttrPtr:
        # Note: while this could generalize to something like splat, MO doesn't
        # really make use of those.
        var shape = DynamicVector[Int](rank)
        for i in range(rank):
            shape.append(1)
        return self.tensor_attr[dtype](name, Tensor(shape, value))

    fn string_attr(self, name: StringRef, value: StringRef) -> AttrPtr:
        return capi.attr_new_string(self.m, name, value)

    # ===------------------------------------------------------------------=== #
    # Graph factories
    # ===------------------------------------------------------------------=== #

    fn graph(self, name: StringRef, in_types: Arity, out_types: Arity) -> Graph:
        let unknown = capi.loc_new_unknown(self.m)
        let g = capi.graph_new(self.m, unknown, name, in_types.a, out_types.a)
        return Graph(g, self)

    # ===------------------------------------------------------------------=== #
    # Type convenience helpers
    # ===------------------------------------------------------------------=== #

    fn i32(self, *dims: Int64) -> TypePtr:
        return MOTensor(DType.int32, dims).to_mlir(self)

    fn i64(self, *dims: Int64) -> TypePtr:
        return MOTensor(DType.int64, dims).to_mlir(self)

    fn f32(self, *dims: Int64) -> TypePtr:
        return MOTensor(DType.float32, dims).to_mlir(self)

    fn bool(self, *dims: Int64) -> TypePtr:
        return MOTensor(DType.bool, dims).to_mlir(self)
