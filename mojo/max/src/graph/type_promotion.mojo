# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines how types can promote in binary operations.

For instance,

```mojo
var g = Graph(...)
let x = g.scalar[DType.int16](1)
let y = g.scalar[DType.float32](1.)
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
counterparts, ie. `float16` can promote to a `bfloat16` but not vice-versa.
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


from collections import Dict, KeyElement, Optional
from collections.set import Set

from max.graph.symbol import Symbol


fn promotion_semilattice() -> Dict[DType, DynamicVector[DType]]:
    """The promotion semilattice structure.

    - Keys of this dictionary are DTypes.
    - Values are lists of DTypes to which the key can be _directly_ promoted.
    - The result will satisfy the semilattice condition for joins, ie.
      for any subset of elements, there is a unique minimal join.

    Returns:
        The promotion semilattice as a dictionary of direct promotions.
    """
    var semilattice = Dict[DType, DynamicVector[DType]]()

    @parameter
    fn promotes_to(dtype: DType, *promotes_to: DType):
        var ps = DynamicVector[DType]()
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

    return semilattice ^


fn min[
    T: KeyElement
](
    semilattice: Dict[T, DynamicVector[T]], owned elements: Set[T]
) raises -> Optional[T]:
    """The smallest given element in the semilattice, if it exists and is
    unique, otherwise None.

    Args:
        semilattice: The semilattice.
        elements: The elements from amongst which to choose the minimum.

    Returns:
        The smallest element in elements, according to the semilattice structure.
        If no unique smallest element exists, None.

    Raises:
        If any element in elements is not in the semilattice.
    """
    var larger = Set[T]()

    for e in elements:
        larger |= semilattice[e[]]

    var smallest = elements - larger
    # min(join(subset)) will be unique as long as semilattice criteria is met
    if len(smallest) != 1:
        return None

    return smallest.pop()


fn all_greater_or_equal_elements[
    T: KeyElement
](semilattice: Dict[T, DynamicVector[T]], element: T) raises -> Set[T]:
    """Find all elements in the partial ordering which are greater than
    or equal to the element, including itself.

    Args:
        semilattice: The semilattice.
        element: The element.

    Returns:
        The set of all elements ordered >= the element.

    Raises:
        - If the element isn't in the semilattice
        - If the semilattice structure doesn't contain keys for
          all its elements.
    """
    var results = Set[T](element)
    var queue = Set[T](element)
    while queue:
        let next = queue.pop()
        # TODO(30973): can't inline this in the for loop
        # TODO(30974): can't bind this with a let
        var es = semilattice[next]
        for e in es:
            if e[] not in results:
                results.add(e[])
                queue.add(e[])

    return results ^


fn join[
    T: KeyElement
](semilattice: Dict[T, DynamicVector[T]], e0: T, *elts: T) raises -> T:
    """The join of any number of elements in the semilattice.

    This is the smallest element which is greater than or equal to
    every passed in element. It is unique as long as the semilattice
    condition holds.

    Args:
        semilattice: The semilattice.
        e0: At least one item to join.
        elts: Any number of other items to join.

    Returns:
        The unique join of the passed in elements.

    Raises:
        - If any passed in element is not in the semilattice
        - If the semilattice doesn't meet the semilattice condition,
          ie. if there is no unique join.
    """
    var gte = all_greater_or_equal_elements(semilattice, e0)
    for e in elts:
        gte &= all_greater_or_equal_elements(semilattice, e[])
    let min_gte = min(semilattice, gte ^)
    if not min_gte:
        raise "No unique join"
    return min_gte.value()


# TODO(30966): Can't make these aliases
# alias PROMOTION_SEMILATTICE = promotion_semilattice()


fn promote(t0: DType, t1: DType) raises -> DType:
    """The smallest DType to which both DTypes may be promoted.

    This gives the correct type to cast two tensors to
    before computing a binary operation.

    Args:
        t0: The first dtype. The order doesn't matter.
        t1: The second dtype. The order doesn't matter.

    Returns:
        The promotion type for a binary operation on two values
        of types t0 and t1 respectively.

    Raises:
        If either DType isn't supported by promotion.
        This should only happen for `DType.invalid`.
    """
    try:
        return join(promotion_semilattice(), t0, t1)
    except:
        raise "No legal promotion for (" + str(t0) + ", " + str(t1) + ")"


fn promote(s0: Symbol, s1: Symbol) raises -> SymbolTuple:
    """Given two graph symbols, promote each of them according
    to the promotion rules.

    See the `max.graph.type_promotion` documentation for more details
    and a full type promotion matrix.

    Cast ops are not added to types which don't need to be promoted.

    Args:
        s0: A graph symbol in a binary operation.
        s1: The other symbol in the binary operation.

    Returns:
        A 2-tuple of the same symbols, possibly cast to their correctly
        promoted type according to the type promotion rules.
    """
    let dtype = promote(
        s0.tensor_type().dtype.dtype, s1.tensor_type().dtype.dtype
    )
    return (s0.cast(dtype), s1.cast(dtype))
