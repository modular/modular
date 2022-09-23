//===- POPOps.td ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the POP dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/KGENDialect/ElaboratorOpInterface.td"
include "KGEN/MetaDialect/MetaTypes.td"
include "KGEN/POPDialect/POPAttrs.td"
include "KGEN/POPDialect/POPDialect.td"
include "KGEN/POPDialect/POPTypes.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class POPOp<string mnemonic, list<Trait> traits = []> :
    Op<POPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def POP_ConstantOp : POPOp<"constant", [ConstantLike, NoSideEffect,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ElaboratorOpInterface, ["finalizeElaboration"]>
  ]> {
  let summary = "integer or floating point constant";
  let description = [{
    The `pop.constant` operation produces a `!meta.scalar` or `!meta.simd` equal
    to the specified integer, floating point, or vector constant. If the
    constant attribute is an integer or floating point, the result type must be
    a scalar. If the constant is a 1D vector elements attribute, the result type
    must be a SIMD vector of the same size.

    If the result dtype is specified, the type of the constant values must be
    convertible from the dtype.

    ```mlir
    // Signless integers can form signed or unsigned integers of the same
    // bitwidth.
    %0 = pop.constant(42 : i64) : !meta.scalar<si64>

    // Explicitly signed or unsigned integers can form only integers of the
    // same bitwidth and signedness.

    %1 = pop.constant(dense<[42, 24]> : vector<2xui64>) : !meta.simd<2, ui64>
    // Floats can only form equivalent floating point.
    %2 = pop.constant(12.3 : f80) : !meta.scalar<f80>
    ```

    If the result dtype is unspecified, the constant values can use integers of
    any size or floats of any precision and semantics:

    ```mlir
    %0 = pop.constant(42) : !meta.scalar<type>
    %1 = pop.constant(12.3 : f32) : !meta.scalar<type>
    %2 = pop.constant(dense<[1, 2]> : vector<2xi64>) : !meta.simd<2, type>
    %3 = pop.constant(dense<[0.1]> : vector<4xbf16>) : !meta.simd<4, type>
    ```

    When the operation is elaborated, the constant is reified according to the
    concretized dtype as follows:

    - If the concretized dtype is floating point, float values are coerced to
      the floating point type if they fit within its range. Integers are
      converted only if they losslessly roundtrip.
    - If the concretized dtype is an integer, integers are converted only if
      truncating or extending (according to the dtype signedness) the value
      leaves it unchanged. Floats are converted only if they are exact integers.

    Examples:

    ```mlir
    // No change in type required.
    %0 = pop.constant(42 : si32) : !meta.scalar<type>
    %0 = pop.constant(42 : si32) : !meta.scalar<si32>

    // `42 : i64` can be represented losslessly as an `f32`.
    %1 = pop.constant(42) : !meta.scalar<type>
    %1 = pop.constant(42.0 : f32) : !meta.scalar<f32>

    // `0.021 : f64` can be coerced to `f32`.
    %1 = pop.constant(0.021) : !meta.scalar<type>
    %1 = pop.constant(0.021 : f32) : !meta.scalar<f32>

    // `16777217 : i32` cannot be represented losslessly as an `f32`. This
    // results in an elaboration error.
    %1 = pop.constant(16777217) : !meta.scalar<type>
    %1 = pop.constant(16777217) : !meta.scalar<f32>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs ScalarOrSIMD:$result);

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(Attribute value, Type type);
  }];

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

// Base class for unary arithmetic pop operations.
class POP_ArithmeticUnaryOp<string mnemonic, list<Trait> traits = []> :
       POPOp<mnemonic, traits # [NoSideEffect, SameOperandsAndResultType]> {
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

// Base class for unary operations.
class POP_MetaArithmeticUnaryOp<string mnemonic, list<Trait> traits = []> :
    POP_ArithmeticUnaryOp<mnemonic, traits # [
                      DeclareOpInterfaceMethods<InferTypeOpInterface>]>,
    Arguments<(ins ScalarOrSIMD:$operand)>,
    Results<(outs ScalarOrSIMD:$result)>;

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

def POP_AbsOp : POP_MetaArithmeticUnaryOp<"abs"> {
  let summary = "absolute value";
  let description = [{
    The `pop.abs` operation takes a meta.scalar or meta.simd type and
    returns its absolute value. The operand and result types must
    be the same.

    Example:

    ```mlir
    %b = pop.abs %a : !meta.scalar<type>
    %b = pop.abs %a : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

def POP_NegOp : POP_MetaArithmeticUnaryOp<"neg"> {
  let summary = "negation";
  let description = [{
    The `pop.neg` operation takes a meta.scalar or meta.simd type and
    returns its negative. The operand and result types must be the same.

    Example:

    ```mlir
    %b = pop.neg %a : !meta.scalar<type>
    %b = pop.neg %a : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

// Base class for binary arithmetic pop operations.
class POP_ArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
       POPOp<mnemonic, traits # [NoSideEffect, SameOperandsAndResultType]> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for binary operations.
class POP_MetaArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
    POP_ArithmeticBinaryOp<mnemonic, traits # [
                      DeclareOpInterfaceMethods<InferTypeOpInterface>]>,
    Arguments<(ins ScalarOrSIMD:$lhs, ScalarOrSIMD:$rhs)>,
    Results<(outs ScalarOrSIMD:$result)>;

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def POP_AddOp : POP_MetaArithmeticBinaryOp<"add", [Commutative]> {
  let summary = "addition";
  let description = [{
    The `pop.add` operation takes two meta.scalar or meta.simd types
    and returns their sum. The operand and result types must be
    the same.

    Example:

    ```mlir
    %c = pop.add %a, %b : !meta.scalar<type>
    %c = pop.add %a, %b : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def POP_SubOp : POP_MetaArithmeticBinaryOp<"sub"> {
  let summary = "subtraction";
  let description = [{
    The `pop.sub` operation takes two meta.scalar or meta.simd types
    and subtracts them. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.sub %a, %b : !meta.scalar<type>
    %c = pop.sub %a, %b : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def POP_MulOp : POP_MetaArithmeticBinaryOp<"mul", [Commutative]> {
  let summary = "multiplication";
  let description = [{
    The `mul` operation takes two meta.scalar or meta.simd types and returns
    their product. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.mul %a, %b : !meta.scalar<type>
    %c = pop.mul %a, %b : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def POP_DivOp : POP_MetaArithmeticBinaryOp<"div"> {
  let summary = "divide";
  let description = [{
    The `div` operation takes two meta.scalar or meta.simd types and divdes
    them. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.div %a, %b : !meta.scalar<type>
    %c = pop.div %a, %b : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def POP_MaxOp : POP_MetaArithmeticBinaryOp<"max", [Commutative]> {
  let summary = "max";
  let description = [{
    The `pop.max` operation takes two meta.scalar or meta.simd types
    and returns their elementwise max. The operand and result types must be
    the same.

    Example:

    ```mlir
    %c = pop.max %a, %b : !meta.scalar<type>
    %c = pop.max %a, %b : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

def POP_MinOp : POP_MetaArithmeticBinaryOp<"min", [Commutative]> {
  let summary = "min";
  let description = [{
    The `pop.min` operation takes two meta.scalar or meta.simd types
    and returns their elementwise min. The operand and result types must be
    the same.

    Example:

    ```mlir
    %c = pop.min %a, %b : !meta.scalar<type>
    %c = pop.min %a, %b : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

def POP_ShlOp : POP_MetaArithmeticBinaryOp<"shl"> {
  let summary = "shift left";
  let description = [{
    The `pop.shl` instruction returns the first operand shifted to the left a
    specified number of bits. Both arguments to the shl instruction must be
    the same integer or vector of integer type. rhs is treated as an unsigned
    value.

    The value produced is lhs * 2^{rhs} mod 2^n, where n is the width of the
    result. If rhs is (statically or dynamically) equal to or larger than the
    number of bits in lhs, this instruction returns a poison value. If the
    arguments are vectors, each vector element of op1 is shifted by the
    corresponding shift amount in op2.

    Example:

    ```mlir
    %0 = pop.shl %val, %shft: !meta.scalar<type>
    %0 = pop.shl %val, %shft: !meta.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIntegerScalarOrSIMD:$lhs,
                       UnboundOrIntegerScalarOrSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// ShrOp
//===----------------------------------------------------------------------===//

def POP_ShrOp : POP_MetaArithmeticBinaryOp<"shr"> {
  let summary = "shift right";
  let description = [{
    The `pop.shr` op returns the first operand shifted to the right a specified
    number of bits. Whether the result is sign extended or zero extended depends
    on the result dtype. Both arguments to the `shr` op must be the same integer
    or vector of integer type. `rhs` is always treated as unsigned.

    This operation performs an arithmetic shift right operation if the dtype
    is signed. The most significant bits of the result will be filled with the
    sign bit of `lhs`. This operation performs a logical shift right operation
    if the dtype is unsigned. The most significant bits of the result will be
    filled with zero bits after the shift.

    If `rhs` is equal to or larger than the number of bits in `lhs`, this op
    returns a poison value. If the arguments are vectors, each vector element of
    `lhs` is shifted by the corresponding shift amount in `rhs`.

    ```mlir
    %0 = pop.shr %val, %shft: !meta.scalar<type>
    %0 = pop.shr %val, %shft: !meta.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIntegerScalarOrSIMD:$lhs,
                       UnboundOrIntegerScalarOrSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

def POP_CopySignOp : POP_MetaArithmeticBinaryOp<"copysign"> {
  let summary = "copy sign";
  let description = [{
    The `pop.copysign` operation takes two meta.scalar or meta.simd types (a
    magnitude `a` and a sign `b`) of floating-point dtype. The `pop.copysign`
    operation copies the sign from `b` to the `a`. The operand and result types
    must be the same and must have a floating-point dtype.

    Example:

    ```mlir
    %c = pop.copysign %a, %b : !meta.scalar<type>
    %c = pop.copysign %a, %b : !meta.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrFloatScalarOrSIMD:$lhs,
                       UnboundOrFloatScalarOrSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

// Base class for ternary arithmetic pop operations.
class POP_ArithmeticTernaryOp<string mnemonic, list<Trait> traits = []> :
       POPOp<mnemonic, traits # [NoSideEffect, SameOperandsAndResultType]> {
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
}

// Base class for ternary operations.
class POP_MetaArithmeticTernaryOp<string mnemonic, list<Trait> traits = []> :
    POP_ArithmeticTernaryOp<mnemonic, traits # [
                      DeclareOpInterfaceMethods<InferTypeOpInterface>]>,
    Arguments<(ins ScalarOrSIMD:$a, ScalarOrSIMD:$b, ScalarOrSIMD:$c)>,
    Results<(outs ScalarOrSIMD:$result)>;

//===----------------------------------------------------------------------===//
// FMAOp
//===----------------------------------------------------------------------===//

def POP_FMAOp : POP_MetaArithmeticTernaryOp<"fma"> {
  let summary = "fused-multiply add";
  let description = [{
    The `pop.fma` operation takes three meta.scalar or meta.simd types (`a`,
    `b`, and `c`) and computes the fma operation `a * b + c`. The operand and
    result types must be the same.

    Example:

    ```mlir
    %d = pop.fma %a, %b, %c : !meta.scalar<type>
    %d = pop.fma %a, %b, %c : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def POP_CmpOp : POPOp<"cmp", [NoSideEffect, SameTypeOperands,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "comparison";
  let description = [{
    The `pop.cmp` operation takes a comparison predicate and two meta.scalar or
    meta.simd types (`lhs` and `rhs`) and computes the resulting comparison
    elementwise.

    The comparison semantics between floating point dtypes is the same as in C.

    Example:

    ```mlir
    %b0 = pop.cmp lt(%a, %b) : !meta.scalar<type>
    %b1 = pop.cmp ge(%c, %d) : !meta.simd<?, type>
    ```
  }];

  let arguments = (ins POP_CmpPredicateAttr:$pred, ScalarOrSIMD:$lhs,
                       ScalarOrSIMD:$rhs);
  let results = (outs BoolScalarOrSIMD:$result);

  let assemblyFormat = "$pred `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}


//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def POP_SelectOp : POPOp<"select",
        [NoSideEffect,
         AllTypesMatch<["trueValue", "falseValue", "result"]>,
         TypesMatchWith<"bool of same branch type", "result", "condition",
                        "getBoolOfSameParentType($_self)">
        ]> {
  let summary = "select";
  let description = [{
    The `pop.select` operation takes a condition predicate and two meta.scalar
    or meta.simd types (`then` and `else`) and chooses one value based on a
    condition, without IR-level branching.

    When the condition is a SIMD vector, then the result is a SIMD vector
    where the value of each element is chosen (aka blended) from the `trueVal`
    and `falseVal` operands based on the condition's element value.

    Example:

    ```mlir
    %a = pop.select %cond, %trueVal, %falseVal : !meta.scalar<type>
    %b = pop.select %cond, %trueVal, %falseVal : !meta.simd<size, type>
    ```
  }];

  let arguments = (ins BoolScalarOrSIMD:$condition, ScalarOrSIMD:$trueValue,
                       ScalarOrSIMD:$falseValue);
  let results = (outs ScalarOrSIMD:$result);

  let assemblyFormat = "operands attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// BitCastOp
//===----------------------------------------------------------------------===//

def POP_BitcastOp : POPOp<"bitcast", [NoSideEffect,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "scalar and simd bitcast";
  let description = [{
    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. The bitcast operation is also used to cast between
    pointer types. When casting between pointer types, both the source and the
    target types must be pointer types.

    Example:

    ```mlir
    %0 = pop.bitcast %a : !meta.scalar<si32> to !meta.scalar<f32>
    %1 = pop.bitcast %b : !meta.scalar<ui64> to !meta.scalar<si64>
    %2 = pop.bitcast %c : !meta.simd<4, f64> to !meta.simd<4, si64>
    %3 = pop.bitcast %d : !meta.simd<2, f64> to !meta.simd<4, f32>
    %4 = pop.bitcast %d : !pop.pointer<f64> to !pop.pointer<!meta.simd<4, f32>>
    ```
  }];

  let arguments = (ins ScalarOrSIMD:$input);
  let results = (outs ScalarOrSIMD:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// PointerBitCastOp
//===----------------------------------------------------------------------===//

def POP_PointerBitcastOp : POPOp<"pointer.bitcast", [NoSideEffect,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "pointer bitcast";
  let description = [{
    The `pop.pointer.bitcast` operation reinterprets a pointer as a pointer to
    a different element type.

    Example:

    ```mlir
    %0 = pop.pointer.bitcast %arg0 : !pop.pointer<?> to
                                     !pop.pointer<!meta.scalar<f32>>

    %1 = pop.pointer.bitcast %0 : !pop.pointer<!meta.scalar<f32>> to
                                  !pop.pointer<!meta.simd<4, f32>>
    ```
  }];

  let arguments = (ins POP_PointerType:$input);
  let results = (outs POP_PointerType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def POP_CastOp : POPOp<"cast", [NoSideEffect]> {
  let summary = "element wise cast";
  let description = [{
    Perform an element-wise cast between SIMD vectors types or between scalar
    types. SIMD vectors must have the same size. The cast semantics are the same
    as C casts. Only bool, integer, and floating point dtypes are supported.

    Example:

    ```mlir
    %0 = pop.cast %a : !meta.scalar<si32> to !meta.scalar<f32>
    %1 = pop.cast %b : !meta.scalar<ui64> to !meta.scalar<si32>
    %2 = pop.cast %c : !meta.simd<4, f64> to !meta.simd<4, ui32>
    ```
  }];

  let arguments = (ins ScalarOrSIMD:$input);
  let results = (outs ScalarOrSIMD:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

def POP_SIMDExtractElementOp : POPOp<"simd.extractelement", [NoSideEffect,
    ScalarOfSameDType<"vector", "result">]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a SIMD vector and an index position and extracts the meta.scalar at
    that position. The position must be of type index.

    Example:

    ```mlir
    %0 = pop.simd.extractelement %0[%position]: !meta.simd<16, f32>
    ```
  }];

  let arguments = (ins SIMDType:$vector, Index:$position);
  let results = (outs ScalarType:$result);

  let assemblyFormat = [{
    $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

def POP_SIMDInsertElementOp : POPOp<"simd.insertelement", [NoSideEffect,
    ScalarOfSameDType<"vector", "value">,
    AllTypesMatch<["result", "vector"]>]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a source value, SIMD vector, and an index position and inserts the
    source value into the vector at the proper position. The position must be
    of type index.

    Example:

    ```mlir
    %0 = pop.simd.extractelement %val, %vec[%position]: !meta.simd<16, f32>
    ```
  }];

  let arguments = (ins SIMDType:$vector, ScalarType:$value, Index:$position);
  let results = (outs SIMDType:$result);

  let assemblyFormat = [{
    $value `,` $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];
}

//===----------------------------------------------------------------------===//
// SIMDShuffle
//===----------------------------------------------------------------------===//

def POP_SIMDShuffleOp : POPOp<"simd.shuffle", [NoSideEffect,
    AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "shufflevector operation";
  let description = [{
    The `pop.simd.shuffle` operation constructs a permutation of two vectors of
    the same length and returns a vector of the same element type with the same
    length as the provided mask, and thus must be a non-parametric constant.

    Example:

    ```mlir
    %0 = pop.simd.shuffle %lhs, %rhs [0, 1, 2, 3]
        : !meta.simd<2, f32> -> !meta.simd<4, f32>
    ```
  }];

  let arguments = (ins SIMDType:$lhs, SIMDType:$rhs, DenseI32ArrayAttr:$mask);
  let results = (outs SIMDType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs $mask attr-dict `:` qualified(type($lhs)) `->`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

def POP_SIMDSplatOp : POPOp<"simd.splat", [NoSideEffect,
    ScalarOfSameDType<"result", "scalar">]> {
  let summary = "SIMD splat";
  let description = [{
    The `pop.simd.splat` operation takes a scalar value and replicates it across
    a SIMD vector and returns that value. For example, splatting `%a` into a
    `vector<4xi32>` will produce a vector valued as `[%a, %a, %a, %a]`.

    ```mlir
    %a = pop.abs %arg0 : !meta.scalar<f32>
    %v = pop.simd.splat %a : !meta.simd<4, f32>

    %u = pop.neg %arg1 : !meta.scalar<type>
    %w = pop.simd.splat %u : !meta.simd<size, type>
    ```

    Note that `!meta.simd<1, type>` is not the same as `!meta.scalar<type>`.
    Hence, this operation will not fold away in those cases.
  }];

  let arguments = (ins ScalarType:$scalar);
  let results = (outs SIMDType:$result);

  let assemblyFormat = "$scalar attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// SIMDReduceOps
//===----------------------------------------------------------------------===//

// Base class for meta.simd reduction operations.
class POP_MetaSIMDReduceOp<string mnemonic, list<Trait> traits = []> :
    POPOp<mnemonic, traits # [ NoSideEffect,
                               ScalarOfSameDType<"operand", "result">]>,
    Arguments<(ins SIMDType:$operand)>,
    Results<(outs ScalarType:$result)> {
  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand))";
}

def POP_SIMDReduceAddOp : POP_MetaSIMDReduceOp<"simd.reduce.add"> {
  let summary = "SIMD reduce add";
  let description = [{
    The `pop.reduce.add` operation takes a simd vector and sums its elements
    into a scalar.

    ```mlir
    %a = pop.reduce.add %arg0 : !meta.simd<size, type>
    ```
  }];
}

def POP_SIMDReduceMulOp : POP_MetaSIMDReduceOp<"simd.reduce.mul"> {
  let summary = "SIMD reduce mul";
  let description = [{
    The `pop.reduce.mul` operation takes a simd vector and multiplies its
    elements into a scalar

    ```mlir
    %a = pop.reduce.mul %arg0 : !meta.simd<size, type>
    ```
  }];
}

def POP_SIMDReduceMaxOp : POP_MetaSIMDReduceOp<"simd.reduce.max"> {
  let summary = "SIMD reduce max";
  let description = [{
    The `pop.reduce.max` operation takes a simd vector and computes the maximum
    element in the vector.

    ```mlir
    %a = pop.reduce.max %arg0 : !meta.simd<size, type>
    ```
  }];
}

def POP_SIMDReduceMinOp : POP_MetaSIMDReduceOp<"simd.reduce.min"> {
  let summary = "SIMD reduce min";
  let description = [{
    The `pop.reduce.min` operation takes a simd vector and computes the minimum
    element in the vector.

    ```mlir
    %a = pop.reduce.min %arg0 : !meta.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def POP_LoadOp : POPOp<"load", [PointerElementType<"ptr", "result">]> {
  let summary = "pointer load operation";
  let description = [{
    Loads the data at the given pointer. Returns a scalar of the same dtype. The
    pointer cannot have a `?` element type. An optional `alignment` attribute
    can be provided to specify the alignment of the load.

    Example:

    ```mlir
    %0 = pop.load %ptr0 : !pop.pointer<!meta.scalar<f32>>
    %1 = pop.load %ptr1 : !pop.pointer<!meta.simd<4, type>>
    %2 = pop.load %ptr0 align 8 : !pop.pointer<!meta.scalar<f32>>
    ```
  }];

  let arguments = (ins Arg<ConcretePointer, "", [MemRead]>:$ptr,
                       OptionalAttr<TypedAttrInterface>:$alignment);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ptr custom<OptionalAlignmentParamValue>($alignment) `` attr-dict `:`
    qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Optional<unsigned>":$alignment)>
  ];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def POP_StoreOp : POPOp<"store", [PointerElementType<"ptr", "arg">]> {
  let summary = "pointer store operation";
  let description = [{
    Store the given scalar data to the given pointer of the same dtype. The
    pointer cannot have a `?` element type. An optional `alignment` attribute
    can be provided to specify the alignment of the store.

    Example:

    ```mlir
    pop.store %0, %ptr0 : !pop.pointer<!meta.scalar<f32>>
    pop.store %1, %ptr1 : !pop.pointer<!meta.simd<4, type>>
    pop.store %1, %ptr1 align 8 : !pop.pointer<!meta.simd<4, type>>
    ```
  }];

  let arguments = (ins AnyType:$arg,
                       Arg<ConcretePointer, "", [MemWrite]>:$ptr,
                       OptionalAttr<TypedAttrInterface>:$alignment);
  let assemblyFormat = [{
    $arg `,` $ptr custom<OptionalAlignmentParamValue>($alignment) `` attr-dict
    `:` qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$arg, "Value":$ptr, "Optional<unsigned>":$alignment)>
  ];
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

def POP_OffsetOp : POPOp<"offset",
    [NoSideEffect, AllTypesMatch<["result", "ptr"]>]> {
  let summary = "pointer offset operation";
  let description = [{
    The `pop.offset` operation takes a pointer and increments or decrements its
    position by a certain amount. The pointer cannot have a `?` element type.

    Example:

    ```mlir
    %addr0 = pop.offset %ptr0[%idx] : !pop.pointer<!meta.scalar<f32>>
    %addr1 = pop.offset %ptr1[%idx] : !pop.pointer<!meta.simd<4, type>>
    ```
  }];

  let arguments = (ins ConcretePointer:$ptr, Index:$index);
  let results = (outs ConcretePointer:$result);
  let assemblyFormat = [{
    $ptr `[` $index `]` attr-dict `:` qualified(type($ptr))
  }];
}

//===----------------------------------------------------------------------===//
// StructConstructOp
//===----------------------------------------------------------------------===//

def POP_StructConstructOp : POPOp<"struct.construct", [NoSideEffect,
    TypesMatchWith<"operand types match struct element types",
                   "result", "elements",
                   "$_self.cast<StructType>().getParameterizedElementTypes()">
]> {
  let summary = "construct a struct";
  let description = [{
    The `pop.struct.construct` operation creates a struct of the given type
    from values corresponding to its element types.

    Example:

    ```mlir
    %0 = pop.struct.construct(%f32, %f64)
      : !pop.struct<!meta.scalar<f32>, !meta.scalar<f64>>

    %1 = pop.struct.construct(%arr, %ptr, %v)
      : !pop.struct<!pop.array<size, type>, !pop.pointer<type>, type>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs POP_StructType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// GetElementOp
//===----------------------------------------------------------------------===//

def POP_GetElementOp : POPOp<"get_element", [NoSideEffect,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "struct extract value";
  let description = [{
    The `pop.get_element` operation gets the struct element at the given
    index from the provided struct.

    Example:

    ```mlir
    // Extract the !meta.scalar<f32> at index 0.
    %0 = pop.get_element %struct[0]
      : !pop.struct<!meta.scalar<f32>, !meta.scalar<f64>>

    // Extract the !meta.scalar<f64> at index 1.
    %1 = pop.get_element %struct[1]
      : !pop.struct<!meta.scalar<f32>, !meta.scalar<f64>>
    ```
  }];

  let arguments = (ins POP_StructType:$container, IndexAttr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $container `[` $index `]` attr-dict `:` qualified(type($container))
  }];

  let builders = [OpBuilder<(ins "Value":$container, "int64_t":$index)>];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReplaceElementOp
//===----------------------------------------------------------------------===//

def POP_ReplaceElementOp : POPOp<"replace_element", [NoSideEffect,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "struct insert value";
  let description = [{
    The `pop.replace_element` operation inserts the given value into the struct
    at the given index. It returns a new struct with the inserted value.

    Example:

    ```mlir
    // Insert the !meta.scalar<f32> at index 0.
    %0 = pop.replace_element %f32, %struct[0]
      : !pop.struct<!meta.scalar<f32>, !meta.scalar<f64>>

    // Insert the !meta.scalar<f64> at index 1.
    %1 = pop.replace_element %f64, %struct[1]
      : !pop.struct<!meta.scalar<f32>, !meta.scalar<f64>>
    ```
  }];

  let arguments = (ins AnyType:$value, POP_StructType:$container,
                       IndexAttr:$index);
  let results = (outs POP_StructType:$result);
  let assemblyFormat = [{
    $value `,` $container `[` $index `]` attr-dict `:`
    qualified(type($container)) ``
    custom<StructValueType>(type($value), ref(type($container)), ref($index))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

def POP_StackAllocationOp : POPOp<"stack_allocation"> {
  let summary = "scoped stack allocation operation";
  let description = [{
    The `pop.stack_allocation` operation allocates statically-sized scoped
    memory on the stack. The operation returns a pointer to the allocated memory
    of the specified size aligned to the dtype of the element type.

    Example:

    ```mlir
    %0 = pop.stack_allocation size : !meta.simd<4, f32>
    %1 = pop.stack_allocation size : !meta.scalar<ui64>
    ```

    The allocated memory is automatically released when control tranfers back
    from the region of its closest surrounding operation.

    Example:

    ```mlir
    kgen.generator @stack_memory<size, type: type>() {
      // This memory is allocated at the start of the function call and released
      // at the end.
      %0 = pop.stack_allocation size : type

      scf.if %cond {
        // This memory is allocated when control flow branches into the 'then'
        // region and released when control flow transfer back.
        %1 = pop.stack_allocation size : type
      }

      scf.for %iv = %lb to %ub step %step {
        // This memory is allocated each time the loop iterates and released
        // at the end of each iteration.
        %2 = pop.stack_allocation size : type
      }
    }
    ```

    If stack memory escapes the scope of its enclosing region, interacting with
    the memory outside its lifetime is undefined behaviour.

    Example:

    ```mlir
    // Scoped to the enclosing region.
    %0 = pop.stack_allocation size : type

    %mem = scf.if %cond -> !pop.pointer<type> {
      // OK: The memory is scoped to the enclosing region.
      scf.yield %0 : !pop.pointer<type>
    } else {
      // Scoped to the 'else' region.
      %1 = pop.stack_allocation size : type

      // The memory is escaping its enclosed region.
      scf.yield %1 : !pop.pointer<type>
    }

    // NOT OK: Loading pointer to invalid memory.
    %v = pop.load %mem : !pop.pointer<type>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$size);
  let results = (outs Res<ConcretePointer, "", [MemAlloc]>:$result);

  let assemblyFormat = [{
    custom<IndexParamValue>($size) attr-dict `:`
    custom<PointerOf>(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// ExternalCallOp
//===----------------------------------------------------------------------===//

def POP_ExternalCallOp : POPOp<"external_call"> {
  let summary = "call an external function";
  let description = [{
    The `pop.external_call` operation invokes an externally-defined function,
    e.g. from libc, using the provided arguments. The function signature is
    inferred from the operand and result types.

    Example:

    ```mlir
    %0 = pop.external_call @sqrt(%arg)
      : (!meta.scalar<f64>) -> !meta.scalar<f32>
    ```

    ```mlir
    %0 = pop.external_call @abs(%arg)
      {type = (!meta.scalar<si32>) -> !meta.scalar<si32>}
      : (!meta.scalar<ui32>) -> !meta.scalar<ui32>
    ```

    Variadic functions are those with trailing function arguments. A variadic
    external function is indicated by specifying an explicit signature of the
    non-variadic arguments.

    ```mlir
    pop.external_call @printf(%fmt, %v0, %v1) (!pop.pointer<si8>) -> ()
      : (!zap.buffer<16, si8>, !meta.scalar<f32>, !meta.scalar<ui32>) -> ()
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$func, Variadic<AnyType>:$operands,
                       OptionalAttr<TypeAttrOf<FunctionType>>:$variadicType);
  let results = (outs Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $func `(` operands `)` ($variadicType^)? attr-dict `:`
    functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// GlobalConstant
//===----------------------------------------------------------------------===//

def POP_GlobalConstantOp : POPOp<"global_constant",
    [NoSideEffect, DeclareOpInterfaceMethods<ElaboratorOpInterface,
                                             ["finalizeElaboration"]>]> {
  let summary = "create a global constant";
  let description = [{
    The `pop.global_constant` operation creates and returns a pointer to a
    global constant value. The returned pointer is read-only. Writing the to
    pointer is undefined behaviour.

    This operation has the same pre and post elaboration constant semantics as
    `pop.constant`.

    Example:

    ```mlir
    %0 = pop.global_constant(5 : i32) : !meta.scalar<ui32>
    %1 = pop.global_constant(12.0) : type
    %2 = pop.global_constant(dense<[4, 2]>) : !meta.simd<2, dtype>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs ConcretePointer:$result);

  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` custom<PointerOf>(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndexToPointerOp
//===----------------------------------------------------------------------===//

def POP_IndexToPointerOp : POPOp<"index_to_pointer", [NoSideEffect]> {
  let summary = "converts an index value to a pointer type.";
  let description = [{
    The `pop.index_to_pointer` instruction takes an index value to cast, and a
    type to cast it to, which must be a pointer type.

    Example:

    ```mlir
    %addr0 = pop.index_to_pointer %idx : !pop.pointer<!meta.scalar<f32>>
    %addr1 = pop.index_to_pointer %idx : !pop.pointer<!meta.simd<4, type>>
    ```
  }];

  let arguments = (ins Index:$value);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

def POP_PointerToIndexOp : POPOp<"pointer_to_index", [NoSideEffect]> {
  let summary = "gets the address of the pointer as an index type.";
  let description = [{
    The `pop.pointer_to_index` instruction takes a pointer to cast and gets
    its address as an index type.

    Example:

    ```mlir
    %addr0 = pop.pointer_to_index %ptr : !pop.pointer<!meta.scalar<f32>>
    %addr1 = pop.pointer_to_index %idx : !pop.pointer<!meta.simd<4, type>>
    ```
  }];

  let arguments = (ins POP_PointerType:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "$value attr-dict `:` qualified(type($value))";
}

//===----------------------------------------------------------------------===//
// TypeLowerOp
//===----------------------------------------------------------------------===//

def POP_TypeLowerOp : POPOp<"type_lower", [NoSideEffect]> {
  let summary = "Unmaterialized type lowering conversion.";
  let description = [{
    The `pop.type_lower` operation represents an as-of-yet unmaterialized
    target type conversion from a high-level type to a low-level type. This
    could be, for instance, lowering from `!meta.scalar` types to their
    corresponding MLIR builtin types, or lowering from `!pop.struct` to
    `!llvm.struct` types. The operation is expected to fold away as the IR is
    lowered.

    The operation acts as an adaptor for mixing low-level, target-specific code
    in KGEN IR.

    Example:

    ```mlir
    kgen.func @llvm_ops(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
      %0 = pop.type_lower %arg0 : !meta.scalar<f32> to f32
      %1 = llvm.fadd %0, %0 : f32
      %2 = pop.type_raise %1 : f32 to !meta.scalar<f32>
      kgen.return %2 : !meta.scalar<f32>
    }
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TypeRaiseOp
//===----------------------------------------------------------------------===//

def POP_TypeRaiseOp : POPOp<"type_raise", [NoSideEffect]> {
  let summary = "Unmaterialized type raising conversion.";
  let description = [{
    The `pop.type_raise` operation represents an as-of-yet unmaterialized
    source type conversion from a low-level type to a high-level type. This
    could be, for instance, lowering from MLIR builtin types to their
    corresponding `!meta.scalar` types, or lowering from `!llvm.struct` to
    `!pop.struct` types. The operation is expected to fold away as the IR is
    lowered.

    The operation acts as an adaptor for mixing low-level, target-specific code
    in KGEN IR.

    Example:

    ```mlir
    kgen.func @llvm_ops(%arg0: !meta.scalar<f32>) -> !meta.scalar<f32> {
      %0 = pop.type_lower %arg0 : !meta.scalar<f32> to f32
      %1 = llvm.fadd %0, %0 : f32
      %2 = pop.type_raise %1 : f32 to !meta.scalar<f32>
      kgen.return %2 : !meta.scalar<f32>
    }
    ```

    Importantly, this operation can adapt parametric constants for use in
    parametric operations.

    Example:

    ```mlir
    kgen.generator @addN<N: si32>(%arg0: !meta.scalar<si32>)
        -> !meta.scalar<si32> {
      %N = kgen.param.constant : si32 = <N>
      %rhs = pop.type_raise %N : si32 to !meta.scalar<si32>
      %result = pop.add %arg0, %rhs : !meta.scalar<f32>
      kgen.return %result : !meta.scalar<f32>
    }
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasFolder = 1;
  let hasVerifier = 1;
}
