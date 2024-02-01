# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import TensorSpec
from utils.variant import Variant

import mlir

from .module import Module


fn _dyn() -> Int64:
    return capi.dim_type_new_dynamic()


@value
struct DynamicDim(CollectionElement):
    pass


@value
struct SymbolicDim(CollectionElement):
    var name: String

    fn __eq__(self, other: SymbolicDim) -> Bool:
        return self.name == other.name


@value
struct StaticDim(CollectionElement):
    var dim: Int64

    fn __init__(inout self, dim: IntLiteral):
        self.dim = dim

    fn __init__(inout self, dim: Int):
        self.dim = dim

    fn __eq__(self, other: StaticDim) -> Bool:
        return self.dim == other.dim


@value
struct Dim(CollectionElement):
    var value: Variant[DynamicDim, StaticDim, SymbolicDim]

    fn __init__(inout self, dim: Int):
        self.value = StaticDim(dim)

    @staticmethod
    fn static(dim: Int64) -> Self:
        return Self(StaticDim(dim))

    @staticmethod
    fn symbolic(name: String) -> Self:
        return Self(SymbolicDim(name))

    @staticmethod
    fn dynamic() -> Self:
        return Self(DynamicDim())

    fn is_dynamic(self) -> Bool:
        return self.value.isa[DynamicDim]()

    fn is_static(self) -> Bool:
        return self.value.isa[StaticDim]()

    fn is_symbolic(self) -> Bool:
        return self.value.isa[SymbolicDim]()

    fn num_elements(self) -> Int64:
        return self.value.get[StaticDim]().dim if self.is_static() else _dyn()

    fn __eq__(self, other: Dim) -> Bool:
        if self.value.isa[DynamicDim]():
            return other.value.isa[DynamicDim]()
        elif self.value.isa[SymbolicDim]():
            return (
                other.value.isa[SymbolicDim]()
                and self.value.get[SymbolicDim]()
                == other.value.get[SymbolicDim]()
            )
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            return (
                other.value.isa[StaticDim]()
                and self.value.get[StaticDim]() == other.value.get[StaticDim]()
            )

    fn __ne__(self, other: Dim) -> Bool:
        return not (self == other)

    fn to_mlir(self, m: Module) -> mlir.Attribute:
        let ctx = m.m.context()
        if self.value.isa[DynamicDim]():
            return capi.dim_new_dynamic(ctx)
        elif self.value.isa[SymbolicDim]():
            let name = self.value.get[SymbolicDim]().name
            let result = capi.dim_new_symbolic(ctx, name._strref_dangerous())
            name._strref_keepalive()
            return result
        else:
            debug_assert(self.value.isa[StaticDim](), "variant cases")
            let dim = self.value.get[StaticDim]().dim
            return capi.dim_new_static(ctx, dim)


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
    var dims: DynamicVector[Dim]
    var ranked: Bool

    # ===------------------------------------------------------------------=== #
    # Constructors
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, dtype: ElementType, *dims: Dim):
        self.dtype = dtype
        self.dims = DynamicVector[Dim](len(dims))
        for d in dims:
            self.dims.append(d[])
        self.ranked = True

    fn __init__(inout self, dtype: ElementType, ranked: Bool):
        self.dtype = dtype
        self.dims = DynamicVector[Dim](0)
        self.ranked = ranked

    fn __init__(inout self, dtype: ElementType, dim: Int):
        # This special overload prevents the DynamicVector overload from
        # stealing precedence due to unpredictable resolution order around
        # implicit casting.
        # The ambiguity is between *Int64 and DynamicVector.
        self.__init__(dtype, Dim(dim))

    fn __init__(inout self, dtype: ElementType, dims: DynamicVector[Dim]):
        self.dtype = dtype
        self.dims = dims
        self.ranked = True

    # ===------------------------------------------------------------------=== #
    # Auxiliary factories
    # ===------------------------------------------------------------------=== #

    fn __init__(inout self, spec: TensorSpec):
        var dims = DynamicVector[Dim](spec.rank())
        for i in range(spec.rank()):
            dims.append(Dim.static(spec[i]))
        self.__init__(spec.dtype(), dims)

    # ===------------------------------------------------------------------=== #
    # MOType trait
    # ===------------------------------------------------------------------=== #

    fn to_mlir(self, m: Module) -> mlir.Type:
        var dims = DynamicVector[mlir.Attribute](len(self.dims))
        for i in range(len(self.dims)):
            dims.append(self.dims[i].to_mlir(m))
        return capi.tensor_type_new(
            m.m,
            self.dtype.to_mlir(m),
            dims,
            self.ranked,
        )

    fn to_string(self, m: Module) -> String:
        return str(self.to_mlir(m))

    @staticmethod
    fn from_mlir(t: mlir.Type) raises -> Self:
        let dtype = capi.tensor_type_get_dtype(t)
        let ranked = capi.tensor_type_is_ranked(t)
        if ranked:
            let rank = capi.tensor_type_get_rank(t)
            var dims = DynamicVector[Dim](rank.to_int())
            for i in range(rank):
                let dim_attr = capi.tensor_type_get_dim(t, i)
                let dim: Dim
                if capi.dim_is_dynamic(dim_attr):
                    dim = Dim.dynamic()
                elif capi.dim_is_static(dim_attr):
                    dim = Dim.static(capi.dim_static_value(dim_attr))
                elif capi.dim_is_symbolic(dim_attr):
                    dim = Dim.symbolic(str(capi.dim_symbolic_name(dim_attr)))
                else:
                    debug_assert(
                        capi.dim_is_symbolic_expression(dim_attr),
                        "unknown dim variant",
                    )
                    raise "Unsupported dim type: symbolic expression"
                dims.push_back(dim)
            return Self(dtype, dims)
        else:
            return Self(dtype, ranked)

    # ===------------------------------------------------------------------=== #
    # Basic accessors
    # ===------------------------------------------------------------------=== #

    fn is_static(self) -> Bool:
        if not self.ranked:
            return False
        for i in range(self.rank()):
            if self.dims[i].is_dynamic():
                return False
        return True

    fn rank(self) -> Int:
        return len(self.dims)

    fn dim(self, pos: Int) raises -> Dim:
        if not self.ranked:
            raise "Cannot get dim of unranked type"
        return self.dims[pos + (self.rank() if pos < 0 else 0)]

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
            if not self.dims[i].is_static():
                return _dyn()
            n *= self.dims[i].num_elements()
        return n


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
