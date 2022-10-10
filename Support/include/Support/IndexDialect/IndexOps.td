//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef INDEXOPS_TD
#define INDEXOPS_TD

include "Support/IndexDialect/IndexEnums.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// IndexOp
//===----------------------------------------------------------------------===//

/// Base class for Index dialect operations.
class IndexOp<string mnemonic, list<Trait> traits = []>
    : Op<IndexDialect, mnemonic, [NoSideEffect] # traits>;

//===----------------------------------------------------------------------===//
// IndexBinaryOp
//===----------------------------------------------------------------------===//

/// Base class for binary Index dialect operations.
class IndexBinaryOp<string mnemonic, list<Trait> traits = []>
    : IndexOp<mnemonic, traits> {
  let arguments = (ins Index:$lhs, Index:$rhs);
  let results = (outs Index:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def Index_AddOp : IndexBinaryOp<"add"> {
  let summary = "index addition";
  let description = [{
    The `add` operation takes two index values and computes their sum.

    Example:

    ```mlir
    // c = a + b
    %c = index.add %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def Index_SubOp : IndexBinaryOp<"sub"> {
  let summary = "index subtraction";
  let description = [{
    The `sub` operation takes two index values and computes the difference of
    the first from the second operand.

    Example:

    ```mlir
    // c = a - b
    %c = index.sub %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def Index_MulOp : IndexBinaryOp<"mul"> {
  let summary = "index multiplication";
  let description = [{
    The `mul` operation takes two index values and computes their product.

    Example:

    ```mlir
    // c = a * b
    %c = index.mul %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivSOp
//===----------------------------------------------------------------------===//

def Index_DivSOp : IndexBinaryOp<"divs"> {
  let summary = "index signed division";
  let description = [{
    The `divs` operation takes two index values and computes their signed
    quotient. Treats the leading bit as the sign and rounds towards zero, i.e.
    `6 / -2 = -3`.

    Note: division by zero and signed division overflow are undefined behaviour.

    Example:

    ```mlir
    // c = a / b
    %c = index.divs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivUOp
//===----------------------------------------------------------------------===//

def Index_DivUOp : IndexBinaryOp<"divu"> {
  let summary = "index unsigned division";
  let description = [{
    The `divu` operation takes two index values and computes their unsigned
    quotient. Treats the leading bit as the most significant and rounds towards
    zero, i.e. `6 / -2 = 0`.

    Note: division by zero is undefined behaviour.

    Example:

    ```mlir
    // c = a / b
    %c = index.divu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CeilDivSOp
//===----------------------------------------------------------------------===//

def Index_CeilDivSOp : IndexBinaryOp<"ceildivs"> {
  let summary = "index signed ceil division";
  let description = [{
    The `ceildivs` operation takes two index values and computes their signed
    quotient. Treats the leading bit as the sign and rounds towards positive
    infinity, i.e. `7 / -2 = -3`.

    Note: division by zero and signed division overflow are undefined behaviour.

    Example:

    ```mlir
    // c = ceil(a / b)
    %c = index.ceildivs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CeilDivUOp
//===----------------------------------------------------------------------===//

def Index_CeilDivUOp : IndexBinaryOp<"ceildivu"> {
  let summary = "index unsigned ceil division";
  let description = [{
    The `ceildivu` operation takes two index values and computes their unsigned
    quotient. Treats the leading bit as the most significant and rounds towards
    positive infinity, i.e. `6 / -2 = 1`.

    Note: division by zero is undefined behaviour.

    Example:

    ```mlir
    // c = ceil(a / b)
    %c = index.ceildivu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FloorDivSOp
//===----------------------------------------------------------------------===//

def Index_FloorDivSOp : IndexBinaryOp<"floordivs"> {
  let summary = "index signed floor division";
  let description = [{
    The `floordivs` operation takes two index values and computes their signed
    quotient. Treats the leading bit as the sign and rounds towards negative
    infinity, i.e. `5 / -2 = -3`.

    Note: division by zero and signed division overflow are undefined behaviour.

    Example:

    ```mlir
    // c = floor(a / b)
    %c = index.floordivs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemSOp
//===----------------------------------------------------------------------===//

def Index_RemSOp : IndexBinaryOp<"rems"> {
  let summary = "index signed remainder";
  let description = [{
    The `rems` operation takes two index values and computes their signed
    remainder. Treats the leading bit as the sign, i.e. `6 % -2 = 0`.

    Example:

    ```mlir
    // c = a % b
    %c = index.rems %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemUOp
//===----------------------------------------------------------------------===//

def Index_RemUOp : IndexBinaryOp<"remu"> {
  let summary = "index unsigned remainder";
  let description = [{
    The `remu` operation takes two index values and computes their unsigned
    remainder. Treats the leading bit as the most significant, i.e.
    `6 % -2 = 6`.

    Example:

    ```mlir
    // c = a % b
    %c = index.remu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxSOp
//===----------------------------------------------------------------------===//

def Index_MaxSOp : IndexBinaryOp<"maxs"> {
  let summary = "index signed maximum";
  let description = [{
    The `maxs` operation takes two index values and computes their signed
    maximum value. Treats the leading bit as the sign, i.e. `max(-2, 6) = 6`.

    Example:

    ```mlir
    // c = max(a, b)
    %c = index.maxs %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxUOp
//===----------------------------------------------------------------------===//

def Index_MaxUOp : IndexBinaryOp<"maxu"> {
  let summary = "index unsigned maximum";
  let description = [{
    The `maxu` operation takes two index values and computes their unsigned
    maximum value. Treats the leading bit as the most significant, i.e.
    `max(-2, 6) = -2`.

    Example:

    ```mlir
    // c = max(a, b)
    %c = index.maxu %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CastSOp
//===----------------------------------------------------------------------===//

def Index_CastSOp : IndexOp<"casts",
                           [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "index signed cast";
  let description = [{
    The `casts` operation enables conversions between values of index type and
    concrete fixed-width integer types. If casting to a wider integer, the value
    is sign-extended. If casting to a narrower integer, the value is truncated.

    Example:

    ```mlir
    // Cast to i32
    %0 = index.casts %a : index to i32

    // Cast from i64
    %1 = index.casts %b : i64 to index
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$input);
  let results = (outs AnyTypeOf<[AnyInteger, Index]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// CastUOp
//===----------------------------------------------------------------------===//

def Index_CastUOp : IndexOp<"castu",
                           [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "index unsigned cast";
  let description = [{
    The `castu` operation enables conversions between values of index type and
    concrete fixed-width integer types. If casting to a wider integer, the value
    is zero-extended. If casting to a narrower integer, the value is truncated.

    Example:

    ```mlir
    // Cast to i32
    %0 = index.castu %a : index to i32

    // Cast from i64
    %1 = index.castu %b : i64 to index
    ```
  }];

  let arguments = (ins AnyTypeOf<[AnyInteger, Index]>:$input);
  let results = (outs AnyTypeOf<[AnyInteger, Index]>:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def Index_CmpOp : IndexOp<"cmp"> {
  let summary = "index compare";
  let description = [{
    The `cmp` operation takes two index values and compares them according to
    the comparison predicate and returns an `i1`. The following comparisons are
    supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    The result is `1` if the comparison is true and `0` otherwise.

    Example:
    ```mlir
    // Signed less than comparison.
    %0 = index.cmp slt(%a, %b)

    // Unsigned greater than or equal comparison.
    %1 = index.cmp uge(%a, %b)

    // Not equal comparison.
    %2 = index.cmp ne(%a, %b)
    ```
  }];

  let arguments = (ins IndexCmpPredicateAttr:$pred, Index:$lhs, Index:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "`` $pred `(` $lhs `,` $rhs `)` attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SizeOfOp
//===----------------------------------------------------------------------===//

def Index_SizeOfOp : IndexOp<"sizeof"> {
  let summary = "size in bits of the index type";
  let description = [{
    The `sizeof` operation produces an index-typed SSA value equal to the sizes
    in bits of the `index` type. For example, on 32-bit systems, the result is
    `32 : index`, and on 64-bit systems, the result is `64 : index`.

    Example:

    ```mlir
    %0 = index.sizeof
    ```
  }];

  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Index_ConstantOp : IndexOp<"constant", [ConstantLike]> {
  let summary = "index constant";
  let description = [{
    The `constant` operation produces an index-typed SSA value equal to some
    integer constant.

    Example:

    ```mlir
    %0 = index.constant 42
    ```
  }];

  let arguments = (ins IndexAttr:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;

  let builders = [OpBuilder<(ins "int64_t":$value)>];
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

def Index_BoolConstantOp : IndexOp<"bool.constant", [ConstantLike]> {
  let summary = "boolean constant";
  let description = [{
    The `bool.constant` operation produces an bool-typed SSA value equal to
    either `true` or `false`.

    This operation is used to materialize bool constants that arise when folding
    `index.cmp`.

    Example:

    ```mlir
    %0 = index.bool.constant true
    ```
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "attr-dict $value";
  let hasFolder = 1;
}

#endif // INDEXOPS_TD
