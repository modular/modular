# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines how types can promote in binary operations.

For instance,

```mojo
var g = Graph(...)
var x = g.scalar[DType.int16](1)
var y = g.scalar[DType.float32](1.)
g.output(x + y)  # what dtype does this have?!
```

We mostly borrow semantics from JAX. We construct our type hierarchy as a lattice,
and the type promotion between two types is their join on the lattice.

Here's a full reference table of the promotion semantics:

| Type           | bool           | int8           | int16          | int32          | int64          | uint8          | uint16         | uint32         | uint64         | index          | address        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|---------|
| bool           | bool           | int8           | int16          | int32          | int64          | uint8          | uint16         | uint32         | uint64         | int64          | int64          | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| int8           | int8           | int8           | int16          | int32          | int64          | int16          | int32          | int64          | float16        | index          | address        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| int16          | int16          | int16          | int16          | int32          | int64          | int16          | int32          | int64          | float16        | float16        | float16        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| int32          | int32          | int32          | int32          | int32          | int64          | int32          | int32          | int64          | float16        | float16        | float16        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| int64          | int64          | int64          | int64          | int64          | int64          | int64          | int64          | int64          | float16        | float16        | float16        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| uint8          | uint8          | int16          | int16          | int32          | int64          | uint8          | uint16         | uint32         | uint64         | uint64         | uint64         | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| uint16         | uint16         | int32          | int32          | int32          | int64          | uint16         | uint16         | uint32         | uint64         | uint64         | uint64         | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| uint32         | uint32         | int64          | int64          | int64          | int64          | uint32         | uint32         | uint32         | uint64         | uint64         | uint64         | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| uint64         | uint64         | float16        | float16        | float16        | float16        | uint64         | uint64         | uint64         | uint64         | uint64         | uint64         | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| index          | uint64         | float16        | float16        | float16        | float16        | uint64         | uint64         | uint64         | uint64         | index          | uint64         | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| address        | uint64         | float16        | float16        | float16        | float16        | uint64         | uint64         | uint64         | uint64         | uint64         | address        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| float16        | float16        | float16        | float16        | float16        | float16        | float16        | float16        | float16        | float16        | float16        | float16        | float16        | bfloat16       | float32        | tensor_float32 | float64 |
| bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | bfloat16       | float32        | tensor_float32 | float64 |
| float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | float32        | tensor_float32 | float64 |
| tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | tensor_float32 | float64 |
| float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64        | float64 |

Some less common dtypes here that might not be familiar:
- `DType.index` is an unsigned word-size integer, like `size_t`.
  It promotes up to uint64 regardless of architecture.
- `DType.address` is an unsigned word-size integer used for pointer types.
  It promotes up to uint64 regardless of architecture.
- `DType.bfloat16` is an alternative float representation. Compared
  to `DType.float16` it moves 3 precision bits into the exponent.
  It promotes up to a `DType.float32`.
- `DType.tensor_float_32` is an alternative float representation. It
  is lower precision than a `DType.float32`. It promotes up to a
  `DType.float64`.

`address` and `index` aren't common and are more a detail of Mojo's DType
implementation, so we never promote other types to them, and they instead
behave as uint64.

`bfloat16` and `tensor_float_32` are placed _above_ their higher-precision
counterparts, in other words, `float16` can promote to a `bfloat16` but not
vice-versa.
This means that unless you're explicitly using `bfloat16` or `tensor_float_32`
you'll never accidentally promote to them, but will always promote to the
higher-precision alternative. If you _are_ using them, be careful to manage
your type promotions correctly. Adding a `bfloat16` to a `float16` will give
a `bfloat16`, for instance. This shouldn't matter in most cases, but
for instance you may lose precision in an operation like `b16 ** f16`,
where `f16` gets cast to a `bfloat16` before the operation and the difference
precision is actually significant to the result, resulting in a lower-precision
output. You can work around this with explicit casts, eg.
`b16.cast(DType.float16) ** f16`.
"""

# TODO: Clean up rules.


from collections import Dict, KeyElement, Optional
from collections.set import Set

from max.graph.symbol import Symbol
from max.graph.ops import cast


fn _promotion_semilattice() -> Dict[DType, List[DType]]:
    """The promotion semilattice structure.

    Properties:
     - Keys of this dictionary are DTypes.
     - Values are lists of DTypes to which the key can be _directly_ promoted.
     - The result will satisfy the semilattice condition for joins, in other
        words for any subset of elements, there is a unique minimal join.

    Returns:
        The promotion semilattice as a dictionary of direct promotions.
    """
    var semilattice = Dict[DType, List[DType]]()

    @parameter
    fn promotes_to(dtype: DType, *promotes_to: DType):
        var ps = List[DType]()
        for p in promotes_to:
            ps.append(p)
        semilattice[dtype] = ps

    promotes_to(DType.bool, DType.int8, DType.uint8)
    promotes_to(DType.int8, DType.int16)
    promotes_to(DType.uint8, DType.int16, DType.uint16)
    promotes_to(DType.int16, DType.int32)
    promotes_to(DType.uint16, DType.int32, DType.uint32)
    promotes_to(DType.int32, DType.int64)
    promotes_to(DType.uint32, DType.int64, DType.uint64)
    promotes_to(DType.index, DType.uint64)
    promotes_to(DType.address, DType.uint64)

    promotes_to(DType.int64, DType.float16)
    promotes_to(DType.uint64, DType.float16)

    promotes_to(DType.bfloat16, DType.float32)
    promotes_to(DType.float16, DType.bfloat16, DType.float32)
    promotes_to(DType.float32, DType.tensor_float32, DType.float64)
    promotes_to(DType.tensor_float32, DType.float64)

    promotes_to(DType.float64)

    return semilattice^


fn _minumum[
    T: KeyElement
](semilattice: Dict[T, List[T]], owned elements: Set[T]) raises -> Optional[T]:
    """Returns the minumum of `elements`, according to the semilattice ordering.

    Args:
        semilattice: The semilattice.
        elements: The elements from amongst which to choose the minimum.

    Returns:
        The smallest value in `elements`. If it is not unique, (in other words,
        the semilattice is not valid), returns `None` instead.

    Raises:
        If any element in elements is not in the semilattice.
    """
    var larger = Set[T]()

    for e in elements:
        larger |= semilattice[e[]]

    var smallest = elements - larger
    # min(_join(subset)) will be unique as long as semilattice criteria is met
    if len(smallest) != 1:
        return None

    return smallest.pop()


fn _greater_subset[
    T: KeyElement
](semilattice: Dict[T, List[T]], element: T) raises -> Set[T]:
    """Returns the subset elements of `semilattice` greater than `element`.

    Args:
        semilattice: The semilattice.
        element: The element.

    Returns:
        The set of all `semilattice` elements greater than `element`.

    Raises:
        If the element not in the semilattice, of if the semilattice is
        inconsistent.
    """
    var results = Set[T](element)
    var queue = Set[T](element)
    while queue:
        var next = queue.pop()
        # TODO(30973): can't inline this in the for loop
        var es = semilattice[next]
        for e in es:
            if e[] not in results:
                results.add(e[])
                queue.add(e[])

    return results^


fn _join[
    T: KeyElement
](semilattice: Dict[T, List[T]], e0: T, *elements: T) raises -> T:
    """Returns the join of `elements`.

    This is the smallest element which is greater than or equal to
    every passed in element. It is unique as long as the semilattice
    condition holds.

    Args:
        semilattice: The semilattice.
        e0: At least one item to _join.
        elements: Any number of other items to _join.

    Returns:
        The unique _join of the passed in elements.

    Raises:
        If any element is not in the semilattice, of if the join is not unique.
    """
    var gte = _greater_subset(semilattice, e0)
    for e in elements:
        gte &= _greater_subset(semilattice, e[])
    var _minumum_gte = _minumum(semilattice, gte^)
    if not _minumum_gte:
        raise "no unique join"
    return _minumum_gte.value()


fn implicit_cast_type(lhs: DType, rhs: DType) raises -> DType:
    """Returns the smallest DType to which both DTypes may be promoted.

    Args:
        lhs: The first dtype.
        rhs: The second dtype.

    Returns:
        The promotion type for a binary operation on two values
        of types lhs and rhs respectively.

    Raises:
        If either DType isn't supported by promotion.
    """
    try:
        return _join(_promotion_semilattice(), lhs, rhs)
    except:
        raise "no legal promotion for (" + str(lhs) + ", " + str(rhs) + ")"


fn implicit_cast(lhs: Symbol, rhs: Symbol) raises -> SymbolTuple:
    """Performs implicit type conversion between operands.

    See the `max.graph.type_promotion` documentation for details on the
    promotion rules.

    Args:
        lhs: The left-hand-side argument.
        rhs: The left-hand-side argument.

    Returns:
        A `SymbolTuple` containing `lhs` and `rhs`, cast to the promoted dtype
        if necessary.
    """
    var dtype = implicit_cast_type(
        lhs.tensor_type().dtype.dtype, rhs.tensor_type().dtype.dtype
    )
    return (cast(lhs, dtype), cast(rhs, dtype))
