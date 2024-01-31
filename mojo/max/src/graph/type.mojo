# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec
from utils.variant import Variant

import mlir

from .module import Module


# TODO: Don't use magic value, return a proper type.
fn dyn() -> Int64:
    return capi.dim_type_new_dynamic()


trait MOType:
    fn to_mlir(self, m: Module) -> mlir.Type:
        ...

    fn to_string(self, m: Module) -> String:
        ...


@value
struct ElementType(MOType):
    # TODO: This mey be insufficient, if we ever need parametric dtypes.
    var dtype: DType

    fn to_mlir(self, m: Module) -> mlir.Type:
        return capi.dtype_new(m.m, self.dtype)

    fn to_string(self, m: Module) -> String:
        return str(self.to_mlir(m))


@value
struct MOTensor(MOType, CollectionElement):
    var dtype: ElementType
    # TODO: This is insufficient. We need a vector of types.
    # To be able to wrap the shape and dims as a type, we need a wrapper over
    # MLIR TypedAttrs first, because MO stores shapes and dims as attributes
    # rather than types.
    var dims: DynamicVector[Int64]
    var ranked: Bool

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, dtype: ElementType, *dims: Int64):
        self.dtype = dtype
        self.dims = DynamicVector[Int64](len(dims))
        for d in dims:
            self.dims.append(d)
        self.ranked = True

    fn __init__(inout self, dtype: ElementType, dims: VariadicList[Int64]):
        self.dtype = dtype
        self.dims = DynamicVector[Int64](len(dims))
        for d in dims:
            self.dims.append(d)
        self.ranked = True

    fn __init__(inout self, dtype: ElementType, ranked: Bool):
        self.dtype = dtype
        self.dims = DynamicVector[Int64](0)
        self.ranked = ranked

    fn __init__(inout self, dtype: ElementType, dim: Int):
        # This special overload prevents the DynamicVector overload from
        # stealing precedence due to unpredictable resolution order around
        # implicit casting.
        # The ambiguity is between *Int64 and DynamicVector.
        self.__init__(dtype, Int64(dim))

    fn __init__(inout self, dtype: ElementType, dims: DynamicVector[Int64]):
        self.dtype = dtype
        self.dims = dims
        self.ranked = True

    # ===------------------------------------------------------------------=== #
    # Auxiliary factories
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, spec: TensorSpec):
        var dims = DynamicVector[Int64](spec.rank())
        for i in range(spec.rank()):
            dims.append(spec[i])
        self.__init__(spec.dtype(), dims)

    # ===------------------------------------------------------------------=== #
    # MOType trait
    # ===------------------------------------------------------------------=== #

    fn to_mlir(self, m: Module) -> mlir.Type:
        return capi.tensor_type_new(
            m.m, self.dtype.to_mlir(m), self.dims, self.ranked
        )

    fn to_string(self, m: Module) -> String:
        return str(self.to_mlir(m))

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn is_static(self) -> Bool:
        if not self.ranked:
            return False
        for i in range(self.rank()):
            if self.dims[i] == dyn():
                return False
        return True

    fn rank(self) -> Int:
        return len(self.dims)

    fn dim(self, pos: Int) raises -> Int64:
        if not self.ranked:
            raise "Cannot get dim of unranked type"
        var i = pos
        if i < 0:
            i = i + self.rank()
        return self.dims[i]

    fn __eq__(self, other: MOTensor) -> Bool:
        if self.dtype.dtype != other.dtype.dtype:
            return False
        if self.ranked != other.ranked:
            return False
        if self.ranked:
            if self.rank() != other.rank():
                return False
            for i in range(self.rank()):
                if self.dims[i] != other.dims[i]:
                    return False
        return True

    # ===------------------------------------------------------------------=== #
    # Shape arithmetic
    # ===------------------------------------------------------------------=== #

    fn num_elements(self) -> Int64:
        var n: Int64 = 1
        for i in range(self.rank()):
            if self.dims[i] == dyn():
                return dyn()
            n *= self.dims[i]
        return n

    # TODO: Add shape arithmetics here, or in a separate Shape type.


@value
struct MOList(MOType, CollectionElement):
    # TODO: This should really be AnyMOType.
    var eltype: MOTensor

    # ===------------------------------------------------------------------=== #
    # MOType trait
    # ===------------------------------------------------------------------=== #

    fn to_mlir(self, m: Module) -> mlir.Type:
        return capi.list_type_new(m.m, self.eltype.to_mlir(m))

    fn to_string(self, m: Module) -> String:
        return str(self.to_mlir(m))


# TODO: Drop this when we get existentials.
alias AnyMOType = Variant[MOTensor, MOList]


# TODO: Drop this when we get existentials.
fn mo_type_to_mlir(t: AnyMOType, m: Module) -> mlir.Type:
    if t.isa[MOTensor]():
        return t.get[MOTensor]().to_mlir(m)
    elif t.isa[MOList]():
        return t.get[MOList]().to_mlir(m)

    # Use a default that will trap, to appease the checker that wants a return.
    return t.get[MOTensor]().to_mlir(m)


@value
struct TypeTuple(Sized):
    var elts: DynamicVector[AnyMOType]

    # ===------------------------------------------------------------------=== #
    # Basic constructors and accessors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, *elts: AnyMOType):
        self.elts = DynamicVector[AnyMOType]()
        for t in elts:
            self.elts.append(t[])

    # ===------------------------------------------------------------------=== #
    # Convenience adapters
    # ===------------------------------------------------------------------=== #

    # TODO: Most should go away when one can express (AnyMOType, AnyMOType)

    fn __init__(inout self, t: MOTensor):
        self.elts = DynamicVector[AnyMOType]()
        self.elts.append(t)

    fn __init__(inout self, t: MOList):
        self.elts = DynamicVector[AnyMOType]()
        self.elts.append(t)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        return len(self.elts)

    # ===------------------------------------------------------------------=== #
    # MLIR conversion
    # ===------------------------------------------------------------------=== #

    fn to_mlir(self, m: Module) -> DynamicVector[mlir.Type]:
        var retval = DynamicVector[mlir.Type]()
        for i in range(len(self.elts)):
            retval.append(mo_type_to_mlir(self.elts[i], m))
        return retval

    # ===------------------------------------------------------------------=== #
    # Mutators
    # ===------------------------------------------------------------------=== #

    fn append(inout self, type: AnyMOType) raises:
        self.elts.append(type)
