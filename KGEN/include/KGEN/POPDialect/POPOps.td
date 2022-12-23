//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the POP dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/KGENDialect/ElaboratorOpInterface.td"
include "KGEN/KGENDialect/KGENAttrs.td"
include "KGEN/POPDialect/POPAttrs.td"
include "KGEN/POPDialect/POPDialect.td"
include "KGEN/POPDialect/POPTypes.td"
include "Support/Interpreter/InterpreterInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class POPOp<string mnemonic, list<Trait> traits = []> :
    Op<POPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def POP_ConstantOp : POPOp<"constant", [ConstantLike, Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<ElaboratorOpInterface, ["finalizeElaboration"]>
  ]> {
  let summary = "integer or floating point constant";
  let description = [{
    The `pop.constant` operation produces a `!pop.scalar` or `!pop.simd` equal
    to the specified integer, floating point, or vector constant. If the
    constant attribute is an integer or floating point, the result type must be
    a scalar. If the constant is a 1D vector elements attribute, the result type
    must be a SIMD vector of the same size.

    If the result dtype is specified, the type of the constant values must be
    convertible from the dtype.

    ```mlir
    // Signless integers can form signed or unsigned integers of the same
    // bitwidth.
    %0 = pop.constant(42 : i64) : !pop.scalar<si64>

    // Explicitly signed or unsigned integers can form only integers of the
    // same bitwidth and signedness.

    %1 = pop.constant(#M.dense_array<42, 24> : vector<2xui64>) : !pop.simd<2, ui64>
    // Floats can only form equivalent floating point.
    %2 = pop.constant(12.3 : f80) : !pop.scalar<f80>
    ```

    If the result dtype is unspecified, the constant values can use integers of
    any size or floats of any precision and semantics:

    ```mlir
    %0 = pop.constant(42) : !pop.scalar<type>
    %1 = pop.constant(12.3 : f32) : !pop.scalar<type>
    %2 = pop.constant(#M.dense_array<1, 2> : vector<2xi64>) : !pop.simd<2, type>
    %3 = pop.constant(#M.dense_array<0.1> : vector<4xbf16>) : !pop.simd<4, type>
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
    %0 = pop.constant(42 : si32) : !pop.scalar<type>
    %0 = pop.constant(42 : si32) : !pop.scalar<si32>

    // `42 : i64` can be represented losslessly as an `f32`.
    %1 = pop.constant(42) : !pop.scalar<type>
    %1 = pop.constant(42.0 : f32) : !pop.scalar<f32>

    // `0.021 : f64` can be coerced to `f32`.
    %1 = pop.constant(0.021) : !pop.scalar<type>
    %1 = pop.constant(0.021 : f32) : !pop.scalar<f32>

    // `16777217 : i32` cannot be represented losslessly as an `f32`. This
    // results in an elaboration error.
    %1 = pop.constant(16777217) : !pop.scalar<type>
    %1 = pop.constant(16777217) : !pop.scalar<f32>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyTypeOf<[KGEN_ParamRefType, POP_SIMDType,
                                 POP_ArrayType]>:$result);

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
      POPOp<mnemonic, traits # [
        Pure,
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<InferTypeOpInterface>
      ]> {
  let arguments = (ins POP_SIMDType:$operand);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

def POP_AbsOp : POP_ArithmeticUnaryOp<"abs"> {
  let summary = "absolute value";
  let description = [{
    The `pop.abs` operation takes a `!pop.scalar` or `!pop.simd` type and
    returns its absolute value. The operand and result types must
    be the same.

    Example:

    ```mlir
    %b = pop.abs %a : !pop.scalar<type>
    %b = pop.abs %a : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

def POP_NegOp : POP_ArithmeticUnaryOp<"neg"> {
  let summary = "negation";
  let description = [{
    The `pop.neg` operation takes a `!pop.scalar` or `!pop.simd` type and
    returns its negative. The operand and result types must be the same.

    Example:

    ```mlir
    %b = pop.neg %a : !pop.scalar<type>
    %b = pop.neg %a : !pop.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIndexOrSignedSIMD:$operand);
  let results = (outs UnboundOrIndexOrSignedSIMD:$result);
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

def POP_CeilOp : POP_ArithmeticUnaryOp<"ceil">  {
  let summary = "ceil";
  let description = [{
    The `pop.ceil` operation takes a `!pop.scalar` or `!pop.simd` type and
    returns its ceiling. The input must have a floating point dtype. The operand
    and result types must be the same.

    This function returns the same values as the libm ceil functions would,
    and handles error conditions in the same way.

    Example:

    ```mlir
    %b = pop.ceil %a : !pop.scalar<type>
    %b = pop.ceil %a : !pop.simd<size, type>
    ```
  }];

  let arguments = (ins UnboundOrFloatSIMD:$operand);
  let results = (outs UnboundOrFloatSIMD:$result);
}

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

// Base class for binary arithmetic pop operations.
class POP_ArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
      POPOp<mnemonic, traits # [
        Pure,
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<InferTypeOpInterface>
      ]> {
  let arguments = (ins POP_SIMDType:$lhs, POP_SIMDType:$rhs);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def POP_AddOp : POP_ArithmeticBinaryOp<"add", [Commutative]> {
  let summary = "addition";
  let description = [{
    The `pop.add` operation takes two `!pop.scalar` or `!pop.simd` types
    and returns their sum. The operand and result types must be
    the same.

    Example:

    ```mlir
    %c = pop.add %a, %b : !pop.scalar<type>
    %c = pop.add %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def POP_SubOp : POP_ArithmeticBinaryOp<"sub"> {
  let summary = "subtraction";
  let description = [{
    The `pop.sub` operation takes two `!pop.scalar` or `!pop.simd` types
    and subtracts them. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.sub %a, %b : !pop.scalar<type>
    %c = pop.sub %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def POP_MulOp : POP_ArithmeticBinaryOp<"mul", [Commutative]> {
  let summary = "multiplication";
  let description = [{
    The `mul` operation takes two `!pop.scalar` or `!pop.simd` types and returns
    their product. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.mul %a, %b : !pop.scalar<type>
    %c = pop.mul %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def POP_DivOp : POP_ArithmeticBinaryOp<"div"> {
  let summary = "divide";
  let description = [{
    The `div` operation takes two `!pop.scalar` or `!pop.simd` types and divdes
    them. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.div %a, %b : !pop.scalar<type>
    %c = pop.div %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RemOp
//===----------------------------------------------------------------------===//

def POP_RemOp : POP_ArithmeticBinaryOp<"rem"> {
  let summary = "remainder";
  let description = [{
    The `pop.rem` operation takes two `!pop.scalar` or `!pop.simd` types and
    returns their remainder. The operand and result types must be the same.

    Example:

    ```mlir
    %c = pop.rem %a, %b : !pop.scalar<type>
    %c = pop.rem %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def POP_MaxOp : POP_ArithmeticBinaryOp<"max", [Commutative]> {
  let summary = "max";
  let description = [{
    The `pop.max` operation takes two `!pop.scalar` or `!pop.simd` types
    and returns their elementwise max. The operand and result types must be
    the same.

    Example:

    ```mlir
    %c = pop.max %a, %b : !pop.scalar<type>
    %c = pop.max %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

def POP_MinOp : POP_ArithmeticBinaryOp<"min", [Commutative]> {
  let summary = "min";
  let description = [{
    The `pop.min` operation takes two `!pop.scalar` or `!pop.simd` types
    and returns their elementwise min. The operand and result types must be
    the same.

    Example:

    ```mlir
    %c = pop.min %a, %b : !pop.scalar<type>
    %c = pop.min %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

def POP_ShlOp : POP_ArithmeticBinaryOp<"shl"> {
  let summary = "shift left";
  let description = [{
    The `pop.shl` instruction returns the first operand shifted to the left a
    specified number of bits. Both arguments to the shl instruction must be
    the same integer/index or vector of integer/index type. rhs is treated as an
    unsigned value.

    The value produced is lhs * 2^{rhs} mod 2^n, where n is the width of the
    result. If rhs is (statically or dynamically) equal to or larger than the
    number of bits in lhs, this instruction returns a poison value. If the
    arguments are vectors, each vector element of op1 is shifted by the
    corresponding shift amount in op2.

    Example:

    ```mlir
    %0 = pop.shl %val, %shft: !pop.scalar<type>
    %0 = pop.shl %val, %shft: !pop.simd<size, type>
    ```
  }];
  let arguments = (ins UnboudOrIndexOrIntegerSIMD:$lhs,
                       UnboudOrIndexOrIntegerSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// ShrOp
//===----------------------------------------------------------------------===//

def POP_ShrOp : POP_ArithmeticBinaryOp<"shr"> {
  let summary = "shift right";
  let description = [{
    The `pop.shr` op returns the first operand shifted to the right a specified
    number of bits. Whether the result is sign extended or zero extended depends
    on the result dtype. Both arguments to the `shr` op must be the same
    integer/index or vector of integer/index type. `rhs` is always treated as
    unsigned.

    This operation performs an arithmetic shift right operation if the dtype
    is signed. The most significant bits of the result will be filled with the
    sign bit of `lhs`. This operation performs a logical shift right operation
    if the dtype is unsigned. The most significant bits of the result will be
    filled with zero bits after the shift.

    If `rhs` is equal to or larger than the number of bits in `lhs`, this op
    returns a poison value. If the arguments are vectors, each vector element of
    `lhs` is shifted by the corresponding shift amount in `rhs`.

    ```mlir
    %0 = pop.shr %val, %shft: !pop.scalar<type>
    %0 = pop.shr %val, %shft: !pop.simd<size, type>
    ```
  }];
  let arguments = (ins UnboudOrIndexOrIntegerSIMD:$lhs,
                       UnboudOrIndexOrIntegerSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// CopySignOp
//===----------------------------------------------------------------------===//

def POP_CopySignOp : POP_ArithmeticBinaryOp<"copysign"> {
  let summary = "copy sign";
  let description = [{
    The `pop.copysign` operation takes two `!pop.scalar` or `!pop.simd` types (a
    magnitude `a` and a sign `b`) of floating-point dtype. The `pop.copysign`
    operation copies the sign from `b` to the `a`. The operand and result types
    must be the same and must have a floating-point dtype.

    Example:

    ```mlir
    %c = pop.copysign %a, %b : !pop.scalar<type>
    %c = pop.copysign %a, %b : !pop.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrFloatSIMD:$lhs,
                       UnboundOrFloatSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

// Base class for ternary arithmetic pop operations.
class POP_ArithmeticTernaryOp<string mnemonic, list<Trait> traits = []> :
      POPOp<mnemonic, traits # [
        Pure,
        SameOperandsAndResultType,
        DeclareOpInterfaceMethods<InferTypeOpInterface>
      ]> {
  let arguments = (ins POP_SIMDType:$a, POP_SIMDType:$b, POP_SIMDType:$c);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:`"
                       " qualified(type($result))";
}
//===----------------------------------------------------------------------===//
// FMAOp
//===----------------------------------------------------------------------===//

def POP_FMAOp : POP_ArithmeticTernaryOp<"fma"> {
  let summary = "fused-multiply add";
  let description = [{
    The `pop.fma` operation takes three `!pop.simd` types (`a`, `b`, and `c`)
    and computes the fma operation `a * b + c`. The operand and result types
    must be the same.

    Example:

    ```mlir
    %d = pop.fma %a, %b, %c : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def POP_CmpOp : POPOp<"cmp", [Pure, SameTypeOperands,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "comparison";
  let description = [{
    The `pop.cmp` operation takes a comparison predicate and two `!pop.scalar`
    or `!pop.simd` types (`lhs` and `rhs`) and computes the resulting comparison
    elementwise.

    The comparison semantics between floating point dtypes is the same as in C.

    Example:

    ```mlir
    %b0 = pop.cmp lt(%a, %b) : !pop.scalar<type>
    %b1 = pop.cmp ge(%c, %d) : !pop.simd<?, type>
    ```
  }];

  let arguments = (ins POP_CmpPredicateAttr:$pred, POP_SIMDType:$lhs,
                       POP_SIMDType:$rhs);
  let results = (outs BoolSIMD:$result);

  let assemblyFormat = "$pred `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// BitwiseOp
//===----------------------------------------------------------------------===//

// Base class for bitwise arithmetic pop operations.
class POP_BitwiseOp<string mnemonic, list<Trait> traits = []> :
      POPOp<mnemonic, traits # [
        Pure,
        Commutative,
        Idempotent,
        SameOperandsAndResultType
      ]> {

  let arguments = (ins UnboundOrBoolOrIndexOrIntegerSIMD:$lhs,
                       UnboundOrBoolOrIndexOrIntegerSIMD:$rhs);
  let results = (outs UnboundOrBoolOrIndexOrIntegerSIMD:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def POP_AndOp : POP_BitwiseOp<"and"> {
  let summary = "bitwise logical and";
  let description = [{
    The `pop.and` returns the bitwise logical and of its two operands. The dtype
    of the operands must be a boolean, integral, or parametric type with
    underlying boolean or integral dtype.

    Example:

    ```mlir
    %c = pop.and %a, %b : !pop.scalar<type>
    %c = pop.and %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def POP_OrOp : POP_BitwiseOp<"or"> {
  let summary = "bitwise logical or";
  let description = [{
    The `pop.or` returns the bitwise logical or of its two operands. The dtype
    of the operands must be a boolean, integral, or parametric type with
    underlying boolean or integral dtype.

    Example:

    ```mlir
    %c = pop.or %a, %b : !pop.scalar<type>
    %c = pop.or %a, %b : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

def POP_XOrOp : POP_BitwiseOp<"xor"> {
  let summary = "bitwise logical xor";
  let description = [{
    The `pop.xor` returns the bitwise logical exclusive or of its two operands.
    The dtype of the operands must be a boolean, integral, or parametric type
    with underlying boolean or integral dtype.

    Example:

    ```mlir
    %c = pop.xor %a, %b : !pop.scalar<type>
    %c = pop.xor %a, %b : !pop.simd<size, type>
    ```

    The XOr is used to implement the `not` operation. For example:

    ```mlir
    %neg_one = pop.constant(-1) : !pop.scalar<type>
    %not = pop.xor %val, %neg_one : !pop.scalar<type> // returns ~(%val)
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def POP_SelectOp : POPOp<"select",
        [Pure,
         AllTypesMatch<["trueValue", "falseValue", "result"]>,
         TypesMatchWith<"bool of same branch type", "result", "condition",
                        "getBoolOfSameParentType($_self)">
        ]> {
  let summary = "select";
  let description = [{
    The `pop.select` operation takes a condition predicate and two `!pop.scalar`
    or `!pop.simd` types (`then` and `else`) and chooses one value based on a
    condition, without IR-level branching.

    When the condition is a SIMD vector, then the result is a SIMD vector
    where the value of each element is chosen (aka blended) from the `trueVal`
    and `falseVal` operands based on the condition's element value.

    Example:

    ```mlir
    %a = pop.select %cond, %trueVal, %falseVal : !pop.scalar<type>
    %b = pop.select %cond, %trueVal, %falseVal : !pop.simd<size, type>
    ```
  }];

  let arguments = (ins BoolSIMD:$condition, POP_SIMDType:$trueValue,
                       POP_SIMDType:$falseValue);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = "operands attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// BitCastOp
//===----------------------------------------------------------------------===//

def POP_BitcastOp : POPOp<"bitcast", [Pure,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "scalar and simd bitcast";
  let description = [{
    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. The bitcast operation should not be used to cast
    between pointer types. For conversion between pointer types one should use
    `pop.pointer.bitcast`.

    Example:

    ```mlir
    %0 = pop.bitcast %a : !pop.scalar<si32> to !pop.scalar<f32>
    %1 = pop.bitcast %b : !pop.scalar<ui64> to !pop.scalar<si64>
    %2 = pop.bitcast %c : !pop.simd<4, f64> to !pop.simd<4, si64>
    %3 = pop.bitcast %d : !pop.simd<2, f64> to !pop.simd<4, f32>
    ```
  }];

  let arguments = (ins POP_SIMDType:$input);
  let results = (outs POP_SIMDType:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `to` "
                       "qualified(type($output))";
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

def POP_PointerBitcastOp : POPOp<"pointer.bitcast", [Pure,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "pointer bitcast";
  let description = [{
    The `pop.pointer.bitcast` operation reinterprets a pointer or function
    pointer as a pointer to a different element or function type.

    Example:

    ```mlir
    // Bitcast an opaque pointer to an f32 pointer.
    %0 = pop.pointer.bitcast %arg0 : !pop.pointer<scalar<invalid>> to
                                     !pop.pointer<scalar<f32>>

    // Bitcast a scalar pointer to a SIMD pointer.
    %1 = pop.pointer.bitcast %0 : !pop.pointer<scalar<f32>> to
                                  !pop.pointer<simd<4, f32>>

    // Bitcast an opaque pointer to a function pointer.
    %2 = pop.pointer.bitcast %arg0 : !pop.pointer<scalar<invalid>> to
                                     (i32, i32) -> (f32, f32)
    ```
  }];

  let arguments = (ins AnyTypeOf<[KGEN_ParamRefType, POP_PointerType,
                                  FunctionType]>:$input);
  let results = (outs AnyTypeOf<[KGEN_ParamRefType, POP_PointerType,
                                 FunctionType]>:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def POP_CastOp : POPOp<"cast", [Pure]> {
  let summary = "element wise cast";
  let description = [{
    Perform an element-wise cast between SIMD vectors types or between scalar
    types. SIMD vectors must have the same size. The cast semantics are the same
    as C casts. Only bool, integer, and floating point dtypes are supported.

    Example:

    ```mlir
    %0 = pop.cast %a : !pop.scalar<si32> to !pop.scalar<f32>
    %1 = pop.cast %b : !pop.scalar<ui64> to !pop.scalar<si32>
    %2 = pop.cast %c : !pop.simd<4, f64> to !pop.simd<4, ui32>
    ```
  }];

  let arguments = (ins POP_SIMDType:$input);
  let results = (outs POP_SIMDType:$output);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `to` "
                       "qualified(type($output))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

def POP_SIMDExtractElementOp : POPOp<"simd.extractelement", [Pure,
    SIMDSizeOneOfSameDType<"vector", "result">]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a SIMD vector and an index position and extracts the `!pop.scalar` at
    that position. The position must be of type index.

    If the position is out-of-bounds, the behavior is undefined.

    Example:

    ```mlir
    %0 = pop.simd.extractelement %0[%position]: !pop.simd<16, f32>
    ```
  }];

  let arguments = (ins POP_SIMDType:$vector, Index:$position);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = [{
    $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

def POP_SIMDInsertElementOp : POPOp<"simd.insertelement", [Pure,
    SIMDSizeOneOfSameDType<"vector", "value">,
    AllTypesMatch<["result", "vector"]>]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a source value, SIMD vector, and an index position and inserts the
    source value into the vector at the proper position. The position must be
    of type index.

    If the position is out-of-bounds, the behavior is undefined.

    Example:

    ```mlir
    %0 = pop.simd.extractelement %val, %vec[%position]: !pop.simd<16, f32>
    ```
  }];

  let arguments = (ins POP_SIMDType:$vector, POP_SIMDType:$value,
                       Index:$position);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = [{
    $value `,` $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];
}

//===----------------------------------------------------------------------===//
// SIMDShuffle
//===----------------------------------------------------------------------===//

def POP_SIMDShuffleOp : POPOp<"simd.shuffle", [Pure,
    AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "shufflevector operation";
  let description = [{
    The `pop.simd.shuffle` operation constructs a permutation of two vectors of
    the same length and returns a vector of the same element type with the same
    length as the provided mask, and thus must be a non-parametric constant.

    If any of the indices is out-of-bounds, the behavior is undefined.

    Example:

    ```mlir
    %0 = pop.simd.shuffle %lhs, %rhs [0, 1, 2, 3]
        : !pop.simd<2, f32> -> !pop.simd<4, f32>
    ```
  }];

  let arguments = (ins POP_SIMDType:$lhs, POP_SIMDType:$rhs,
                       DenseI32ArrayAttr:$mask);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs $mask attr-dict `:` qualified(type($lhs)) `->`
    qualified(type($result))
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

def POP_SIMDSplatOp : POPOp<"simd.splat", [Pure,
    SIMDSizeOneOfSameDType<"result", "scalar">]> {
  let summary = "SIMD splat";
  let description = [{
    The `pop.simd.splat` operation takes a scalar value and replicates it across
    a SIMD vector and returns that value. For example, splatting `%a` into a
    `vector<4xi32>` will produce a vector valued as `[%a, %a, %a, %a]`.

    ```mlir
    %a = pop.abs %arg0 : !pop.simd<1, f32>
    %v = pop.simd.splat %a : !pop.simd<4, f32>

    %u = pop.neg %arg1 : !pop.simd<1, type>
    %w = pop.simd.splat %u : !pop.simd<size, type>
    ```
  }];

  let arguments = (ins POP_SIMDType:$scalar);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = "$scalar attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// SIMDReduceOps
//===----------------------------------------------------------------------===//

// Base class for SIMD reduction operations.
class POP_MetaSIMDReduceOp<string mnemonic, list<Trait> traits = []> :
    POPOp<mnemonic, traits # [Pure,
                              SIMDSizeOneOfSameDType<"operand", "result">]>,
    Arguments<(ins UnboudOrIndexOrArithmeticSIMD:$operand)>,
    Results<(outs UnboudOrIndexOrArithmeticSIMD:$result)> {
  let assemblyFormat = "$operand attr-dict `:` qualified(type($operand))";
}

def POP_SIMDReduceAddOp : POP_MetaSIMDReduceOp<"simd.reduce.add"> {
  let summary = "SIMD reduce add";
  let description = [{
    The `pop.reduce.add` operation takes a simd vector and sums its elements
    into a scalar.

    ```mlir
    %a = pop.reduce.add %arg0 : !pop.simd<size, type>
    ```
  }];
}

def POP_SIMDReduceMulOp : POP_MetaSIMDReduceOp<"simd.reduce.mul"> {
  let summary = "SIMD reduce mul";
  let description = [{
    The `pop.reduce.mul` operation takes a simd vector and multiplies its
    elements into a scalar

    ```mlir
    %a = pop.reduce.mul %arg0 : !pop.simd<size, type>
    ```
  }];
}

def POP_SIMDReduceMaxOp : POP_MetaSIMDReduceOp<"simd.reduce.max"> {
  let summary = "SIMD reduce max";
  let description = [{
    The `pop.reduce.max` operation takes a simd vector and computes the maximum
    element in the vector.

    ```mlir
    %a = pop.reduce.max %arg0 : !pop.simd<size, type>
    ```
  }];
}

def POP_SIMDReduceMinOp : POP_MetaSIMDReduceOp<"simd.reduce.min"> {
  let summary = "SIMD reduce min";
  let description = [{
    The `pop.reduce.min` operation takes a simd vector and computes the minimum
    element in the vector.

    ```mlir
    %a = pop.reduce.min %arg0 : !pop.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SIMDGatherOp
//===----------------------------------------------------------------------===//

def POP_SIMDGatherOp : POPOp<"simd.gather", [
    AllTypesMatch<["result", "passthrough"]>,
    AddressSIMDOfSameSize<"passthrough", "base">,
    BoolSIMDOfSameSize<"passthrough", "mask">]> {
  let summary = "gathers values from the SIMD vector using the mask";
  let description = [{
    The `pop.simd.gather` reads scalar values from a SIMD vector of memory
    locations and gathers them into one vector. The memory locations are
    provided in the vector of pointers `base` as addresses. The memory is
    accessed according to the provided mask. The mask holds a bit for each
    vector lane, and is used to prevent memory accesses to the masked-off
    lanes. The masked-off lanes in the result vector are taken from the
    corresponding lanes of the `passthrough` operand.

    The `pop.simd.gather` is designed for conditional reading of multiple
    scalar values from arbitrary memory locations in a single IR operation. It
    is useful for targets that support vector masked gathers and allows
    vectorizing basic blocks with data and control divergence. Other targets
    may support this intrinsic differently, for example by lowering it into a
    sequence of scalar load operations. The semantics of this operation are
    equivalent to a sequence of conditional scalar loads with subsequent
    gathering all loaded values into a single vector. The mask restricts memory
    access to certain lanes and facilitates vectorization of predicated basic
    blocks.

    Assuming that %base is a vector of pointers [p0, p1, p2, p3] and %mask is a
    vector of i1 values [true, false, true, false], and the passthrough vector
    is [d0, d1, d2, d3] then the following op:

    ```mlir
    %0 = pop.simd.gather %base[%mask], %passthrough : !pop.simd<4, type>
    ```

    is equivalent to the following sequence of scalar loads:

    ```mlir
    %0 = load %p0 : !pop.scalar<type>
    %1 = load %p2 : !pop.scalar<type>
    %2 = pop.constant(0) : !pop.simd<4, type>
    %3 = pop.simd.insertelement %0, %2[0] : !pop.simd<4, type>
    %4 = pop.simd.insertelement %d0, %3[1] : !pop.simd<4, type>
    %5 = pop.simd.insertelement %1, %4[2] : !pop.simd<4, type>
    %6 = pop.simd.insertelement %d3, %5[2] : !pop.simd<4, type>
    ```

    In general, for some vector of pointers %base, mask %mask, and passthrough
    %pass instructions of the form:

    ```mlir
    %0 = pop.simd.gather %base[%mask], %passthrough : !pop.simd<N, type>
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      result[i] = mask[i] ? *base[i] : passthrough[i];
    ```
  }];

  let arguments = (ins Arg<AddressSIMD, "", [MemRead]>:$base,
                       BoolSIMD:$mask,
                       POP_SIMDType:$passthrough);
  let results = (outs POP_SIMDType:$result);

  let assemblyFormat = [{
    $base `[` $mask `]` `,` $passthrough attr-dict `:`
    qualified(type($passthrough))
  }];
}

//===----------------------------------------------------------------------===//
// SIMDScatterOp
//===----------------------------------------------------------------------===//

def POP_SIMDScatterOp : POPOp<"simd.scatter", [
    AddressSIMDOfSameSize<"value", "base">,
    BoolSIMDOfSameSize<"value", "mask">]> {
  let summary = "scatters values from the SIMD vector using the mask";
  let description = [{
    The `pop.simd.scatter` stores scalar values from a SIMD vector of memory
    locations and scatters them into a vector of pointers. The memory locations
    are provided in the vector of pointers `base` as addresses. The memory is
    stored according to the provided mask. The mask holds a bit for each vector
    lane, and is used to prevent memory accesses to the masked-off lanes.

    The `value` operand is a vector value to be written to memory. The `base`
    operand is a vector of pointers, pointing to where the value elements
    should be stored. It has the same underlying type as the value operand. The
    `mask` operand, mask, is a vector of boolean values. The types of the
    `mask` and the `value` operand must have the same number of vector
    elements.

    The behavior of the `pop.simd.scatter` is undefined if the op stores into
    the same memory location more than once.

    In general, for some vector %value, vector of pointers %base, and mask
    %mask instructions of the form:

    ```mlir
    %0 = pop.simd.gather %value, %base[%mask] : !pop.simd<N, type>
    ```

    is equivalent to the following sequence of scalar loads in C++:

    ```cpp
    for (int i = 0; i < N; i++)
      if (mask[i])
        base[i] = value[i];
    ```
  }];

  let arguments = (ins POP_SIMDType:$value,
                       Arg<AddressSIMD, "", [MemWrite]>:$base,
                       BoolSIMD:$mask);

  let assemblyFormat = [{
    $value `,` $base `[` $mask `]` attr-dict `:` qualified(type($value))
  }];
}


//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def POP_LoadOp : POPOp<"load", [PointerElementType<"ptr", "result">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pointer load operation";
  let description = [{
    Loads the data at the given pointer. Returns a scalar of the same dtype. The
    pointer cannot have a `?` element type. An optional `align` attribute can be
    provided to specify the alignment of the load.

    Example:

    ```mlir
    %0 = pop.load %ptr0 : !pop.pointer<scalar<f32>>
    %1 = pop.load %ptr1 : !pop.pointer<simd<4, type>>
    %2 = pop.load %ptr0 align 8 : !pop.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins Arg<POP_PointerType, "", [MemRead]>:$ptr,
                       OptionalAttr<TypedAttrInterface>:$alignment);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ptr `` custom<OptionalAlignmentParamValue>($alignment) attr-dict `:`
    qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$ptr, "Optional<unsigned>":$alignment)>,
    OpBuilder<(ins "Value":$ptr, "TypedAttr":$alignment)>
  ];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def POP_StoreOp : POPOp<"store", [PointerElementType<"ptr", "arg">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pointer store operation";
  let description = [{
    Store the given scalar data to the given pointer of the same dtype. The
    pointer cannot have a `?` element type. An optional `alignment` attribute
    can be provided to specify the alignment of the store.

    Example:

    ```mlir
    pop.store %0, %ptr0 : !pop.pointer<scalar<f32>>
    pop.store %1, %ptr1 : !pop.pointer<simd<4, type>>
    pop.store %1, %ptr1 align 8 : !pop.pointer<simd<4, type>>
    ```
  }];

  let arguments = (ins AnyType:$arg,
                       Arg<POP_PointerType, "", [MemWrite]>:$ptr,
                       OptionalAttr<TypedAttrInterface>:$alignment);
  let assemblyFormat = [{
    $arg `,` $ptr `` custom<OptionalAlignmentParamValue>($alignment) attr-dict
    `:` qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$arg, "Value":$ptr, "Optional<unsigned>":$alignment)>
  ];
}

//===----------------------------------------------------------------------===//
// PrefetchOp
//===----------------------------------------------------------------------===//

def POP_PrefetchOp : POPOp<"prefetch"> {
  let summary = "prefetch instruction or data into cache";
  let description = [{
    The `pop.prefetch` operation provides prefetching hints for the target
    to prefetch instruction or data into cache before they are used.

    This op follows similar interface as prefetch op in LLVM dialect,
    with a "locality" attribute that specifies the level of temporal locality
    in the application, i.e. how soon would the same data be visited again,
    possible locality values are: `NoLocality`, `LowLocality`, `MediumLocality`,
    `HighLocality`, and `VeryHighLocality`.

    This op also takes a "cache tag" attribute giving hints on how the
    prefetched data will be used. Possible tags are: `ReadICache`, `ReadDCache`
    and `WriteDCache`.

    Note: the actual behavior of this op and concrete interpretation
    of these attributes are target-dependent.

    Example:

    ```mlir
    pop.prefetch %p (NoLocality, ReadDCache) : !pop.pointer<scalar<f32>>
    pop.prefetch %p (HighLocality, WriteDCache) : !pop.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins POP_PointerType:$ptr,
                       POP_PrefetchLocalityAttr:$locality,
                       POP_PrefetchTagAttr:$cache_tag);

  let assemblyFormat = [{
    $ptr `(` $locality `,` $cache_tag `)` attr-dict `:` qualified(type($ptr))
  }];
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

def POP_OffsetOp : POPOp<"offset",
    [Pure, AllTypesMatch<["result", "ptr"]>]> {
  let summary = "pointer offset operation";
  let description = [{
    The `pop.offset` operation takes a pointer and increments or decrements its
    position by a certain amount. The pointer cannot have a `?` element type.

    Example:

    ```mlir
    %addr0 = pop.offset %ptr0[%idx] : !pop.pointer<scalar<f32>>
    %addr1 = pop.offset %ptr1[%idx] : !pop.pointer<simd<4, type>>
    ```
  }];

  let arguments = (ins POP_PointerType:$ptr, Index:$index);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = [{
    $ptr `[` $index `]` attr-dict `:` qualified(type($ptr))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructConstructOp
//===----------------------------------------------------------------------===//

def POP_StructConstructOp : POPOp<"struct.construct", [Pure,
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
      : !pop.struct<scalar<f32>, scalar<f64>>

    %1 = pop.struct.construct(%arr, %ptr, %v)
      : !pop.struct<array<size, type>, pointer<type>, type>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs POP_StructType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` qualified(type($result))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructGetOp
//===----------------------------------------------------------------------===//

def POP_StructGetOp : POPOp<"struct.get", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "struct extract value";
  let description = [{
    The `pop.struct.get` operation gets the struct element at the given
    index from the provided struct.

    Example:

    ```mlir
    // Extract the !pop.scalar<f32> at index 0.
    %0 = pop.struct.get %struct[0]
      : !pop.struct<scalar<f32>, scalar<f64>>

    // Extract the !pop.scalar<f64> at index 1.
    %1 = pop.struct.get %struct[1]
      : !pop.struct<scalar<f32>, scalar<f64>>
    ```
  }];

  let arguments = (ins POP_StructType:$container, IndexAttr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $container `[` $index `]` attr-dict `:` qualified(type($container))
  }];

  let builders = [OpBuilder<(ins "Value":$container, "int64_t":$index)>];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

def POP_StructReplaceOp : POPOp<"struct.replace", [Pure,
    AllTypesMatch<["container", "result"]>]> {
  let summary = "struct insert value";
  let description = [{
    The `pop.struct.replace` operation inserts the given value into the struct
    at the given index. It returns a new struct with the inserted value.

    Example:

    ```mlir
    // Insert the !pop.scalar<f32> at index 0.
    %0 = pop.struct.replace %f32, %struct[0]
      : !pop.struct<scalar<f32>, scalar<f64>>

    // Insert the !pop.scalar<f64> at index 1.
    %1 = pop.struct.replace %f64, %struct[1]
      : !pop.struct<scalar<f32>, scalar<f64>>
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
  let hasFolder = 1;

  let builders = [
    OpBuilder<(ins "Value":$value, "Value":$container, "int64_t":$index)>
  ];
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

def POP_StructGEPOp : POPOp<"struct.gep", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "struct get element pointer";
  let description = [{
    The `pop.struct.gep` operation takes a pointer to a `!pop.struct` and a
    constant index and returns a pointer to the struct element at that index.

    Example:

    ```mlir
    %struct = pop.stack_allocation 1 : !pop.struct<i32, i64>
    %i64Ptr = pop.struct.gep %struct[1] : <struct<i32, i64>>
    ```

    To get a pointer to a struct from a value-semantic struct, the struct value
    must be stored into a block of allocated memory first.

    ```mlir
    ^bb0(%struct: !pop.struct<i32, i64>):
      %mem = pop.stack_allocation 1 : !pop.struct<i32, i64>
      pop.store %struct, %mem : !pop.pointer<struct<i32, i64>>
    ```
  }];

  let arguments = (ins PointerOf<POP_StructType>:$container, IndexAttr:$index);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = [{
    $container `[` $index `]` attr-dict `:` type($container)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

def POP_ArrayCreateOp : POPOp<"array.create", [Pure,
    ArrayElementType<"result", "operands">]> {
  let summary = "create an array";
  let description = [{
    The `pop.array.create` operation creates an `!pop.array` populated with the
    given SSA values. The resulting array must have a fixed size equal to the
    number of operands.

    Example:

    ```mlir
    // Create an array of scalar floats.
    %0 = pop.array.create [%arg0, %arg1] : !pop.array<2, scalar<f32>>

    // Create an array of structs.
    %1 = pop.array.create [%arg2, %arg3] : !pop.array<2, struct<>>

    // Create an array of parametric types.
    %2 = pop.array.create [%arg4] : !pop.array<1, T>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs ArrayKnownSize:$result);
  let assemblyFormat = [{
    ` ` `[` $operands `]` attr-dict `:` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$elements)>
  ];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayRepeatOp
//===----------------------------------------------------------------------===//

def POP_ArrayRepeatOp : POPOp<"array.repeat", [Pure,
    ArrayElementType<"result", "operands">]> {
  let summary = "create an array";
  let description = [{
    The `pop.array.repeat` operation facilitates the creation of splat or
    parametric arrays. The operation repeats the given operands into the
    resulting array in the provided order until the array is full. Extra
    elements are truncated.

    It is only valid to have no operands if the resulting array has no elements.

    Example:

    ```mlir
    // Create a splat array.
    %0 = pop.array.repeat [%arg0] : !pop.array<N, scalar<f32>>

    // Repeat the pair to fill the array. The last element is dropped.
    %1 = pop.array.repeat [%arg2, %arg3] : !pop.array<9, T>

    // Repeat the triplet to populate the array.
    %2 = pop.array.create [%a, %b, %c] : !pop.array<M, T>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs ArrayKnownSize:$result);
  let assemblyFormat = [{
    ` ` `[` $operands `]` attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

def POP_ArrayGetOp : POPOp<"array.get", [Pure,
    ArrayElementType<"array", "result">]> {
  let summary = "array get element";
  let description = [{
    The `pop.array.get` operation extracts the element of the given array at the
    specified index. An out-of-bounds index is invalid.

    Example:

    ```mlir
    // Get the third element.
    %0 = pop.array.get %arr0[2] : !pop.array<4, T>

    // Get the first element.
    %1 = pop.array.get %arr1[0] : !pop.array<N, f32>
    ```
  }];

  let arguments = (ins POP_ArrayType:$array, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $array `[` custom<IndexParamValue>($index) `]` attr-dict `:`
    qualified(type($array))
  }];

  let builders = [OpBuilder<(ins "Value":$array, "int64_t":$index)>];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

def POP_ArrayReplaceOp : POPOp<"array.replace", [Pure,
    AllTypesMatch<["array", "result"]>, ArrayElementType<"result", "value">]> {
  let summary = "array replace element";
  let description = [{
    The `pop.array.replace` operation inserts the given into the array at the
    given index. It returns a new array with the inserted value. The original
    array is not modified. An out-of-bounds index is invalid.

    Example:

    ```mlir
    // Replace the second element.
    %0 = pop.array.replace %a, %arr0[1] : !pop.array<4, T>

    // Replace the first element.
    %1 = pop.array.replace %b, %arr1[0] : !pop.array<N, i32>
    ```
  }];

  let arguments = (ins AnyType:$value, POP_ArrayType:$array,
                       AttrOfType<Index>:$index);
  let results = (outs POP_ArrayType:$result);
  let assemblyFormat = [{
    $value `,` $array `[` custom<IndexParamValue>($index) `]` attr-dict `:`
    qualified(type($result))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

def POP_ArrayGEPOp : POPOp<"array.gep", [Pure,
    TypesMatchWith<"pointer to array element type", "array", "result",
                   "getPointerToArrayElementType($_self)">]> {
  let summary = "array get element pointer";
  let description = [{
    The `pop.array.gep` operation takes a pointer to a `!pop.array` and an index
    and returns a pointer to the array element at that index. The behaviour is
    undefined if the index is out-of-bounds.

    Example:

    ```mlir
    %arr = pop.stack_allocation 1 : !pop.array<4, i32>
    %two = index.constant 2
    %v2 = pop.array.gep %arr[%two] : <array<4, i32>>
    ```
  }];

  let arguments = (ins PointerOf<POP_ArrayType>:$array, Index:$index);
  let results = (outs POP_PointerType:$result);
  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array)";
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

def VariantCreateOp : POPOp<"variant.create", [Pure]> {
  let summary = "Create a parametric variant.";
  let description = [{
    The `pop.variant.create` operation creates a variant of the referred type
    with a value provided for one of its possible types.

    Example:

    ```mlir
    // Create an `Optional<T>` variant.
    %none = pop.struct.construct() : !pop.struct<>
    %0 = pop.variant.create %none : !pop.struct<>
        -> !pop.variant<struct<>, T>

    // Create a variant of either a scalar float or integer.
    %1 = pop.constant(0.0) : !pop.scalar<f64>
    %2 = pop.variant.create %1 : !pop.scalar<f64>
        -> !pop.variant<scalar<i64>, scalar<f64>>
    ```
  }];

  let arguments = (ins AnyType:$operand);
  let results = (outs POP_VariantType:$result);
  let assemblyFormat = [{
    $operand attr-dict `:` type($operand) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

def VariantIsOp : POPOp<"variant.is", [Pure]> {
  let summary = "Check the field contained by a variant.";
  let description = [{
    The `pop.variant.is` operation checks whether the given variant contains
    a particular type. Returns an `i1` that indicates whether the variant is the
    particular type.

    Example:

    ```mlir
    %optional = ...
    %isNone = pop.variant.is !pop.struct<>, %optional
        : !pop.variant<struct<>, T>
    ```
  }];

  let arguments = (ins POP_VariantType:$variant, TypeAttr:$testType);
  let results = (outs I1:$result);
  let assemblyFormat = [{
    $testType `,` $variant attr-dict `:` qualified(type($variant))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

def VariantGetOp : POPOp<"variant.get", [Pure]> {
  let summary = "Get a variant as one of its possible types.";
  let description = [{
    The `pop.variant.get` operation interprets the given variant as one of its
    possible types and gets it as that type. This operation does NOT check
    whether the variant is that type. If the result type does not match the
    actual type of the variant, the operation's result is undefined.

    Example:

    ```mlir
    %optional = ...
    %intVal = pop.variant.get %optional
        : !pop.variant<struct<>, i32> as i32
    ```
  }];

  let arguments = (ins POP_VariantType:$variant);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $variant attr-dict `:` qualified(type($variant)) `as` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantVisitOp
//===----------------------------------------------------------------------===//

def POP_VariantVisitOp : POPOp<"variant.visit", [RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface, [
      "getSuccessorEntryOperands", "getRegionInvocationBounds",
      "areTypesCompatible"]>]> {
  let summary = "variant case visit";
  let description = [{
    The `pop.variant.visit` operation performs a control-flow switch on the
    type of the variant, visiting the appropriate region based on the type of
    the variant. The operation can return any number of results, but each region
    must region the same results.

    Either a case must be specified for each possible type OR a case can be
    specified for a subset of possible types with a default region. Duplicate
    cases or case types that are not one of the variant's types are invalid.

    Example:

    ```mlir
    // Switch on the type of a variant. Return an index.
    %0 = pop.variant.switch %arg0 : !pop.variant<i32, f32> -> index
    case i32 {
      %one = index.constant 1
      pop.yield %one : index
    }
    case f32 {
      %two = index.constant 2
      pop.yield %two : index
    }

    // Switch on the type of a parametric variant. Default case.
    pop.variant.switch %arg1 : !pop.variant<A, B, C>
    case A {
      zap.print "A!\n"()
    }
    default {}
    ```
  }];

  let arguments = (ins POP_VariantType:$variant, KGEN_TypeArrayAttr:$cases);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let assemblyFormat = [{
    $variant attr-dict `:` qualified(type($variant)) (`->` type($results)^)?
    custom<VariantVisitRegions>($cases, $regions)
  }];

  let extraClassDeclaration = [{
    /// Returns true if the operation has a default region.
    bool hasDefaultRegion();
    /// Return the default region.
    Region *getDefaultRegion();
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def POP_YieldOp : POPOp<"yield", [Pure, ReturnLike, Terminator,
    HasParent<"VariantVisitOp">]> {
  let summary = "yield operation";
  let description = [{
    The `pop.yield` operation is a terminator for control-flow regions that
    transfers execution and results back to the parent operation.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict (`:` type($operands)^)?";
}

//===----------------------------------------------------------------------===//
// ListGetOp
//===----------------------------------------------------------------------===//

def POP_ListGetOp : POPOp<"list.get", [Pure,
    TypesMatchWith<"list element type", "list", "result",
    "ParamRefType::get($_self.cast<ListType>().getElementType())">]> {
  let summary = "Get a list element.";
  let description = [{
    The `pop.list.get` operation returns the element of a list at the provided
    index. The index can be an expression, but if it resolves to an out-of-range
    index, an error is emitted.

    Example:

    ```mlir
    %0 = pop.list.get %list[add(A, N)] : <index[N]>
    ```
  }];

  let arguments = (ins KGEN_ListType:$list, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $list `[` custom<IndexParamValue>($index) `]` attr-dict `:` type($list)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ListCreateOp
//===----------------------------------------------------------------------===//

def POP_ListCreateOp : POPOp<"list.create", [Pure,
    TypesMatchWith<"list element type", "result", "operands",
      "ParamRefType::get($_self.cast<ListType>().getElementType())",
      "typeRangeMatches">]> {
  let summary = "Form a list.";
  let description = [{
    The `pop.list.create` operation forms a list using the provided operands.

    Example:

    ```mlir
    %0 = pop.list.create(%arg0, %arg1) : <index[2]>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs KGEN_ListType:$result);
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// IndirectCallOp
//===----------------------------------------------------------------------===//

def POP_IndirectCallOp : POPOp<"indirect_call", [
    TypesMatchWith<"operand types match callee argument types", "callee",
                   "arguments", "$_self.cast<FunctionType>().getInputs()">,
    TypesMatchWith<"result types match callee result types", "callee",
                   "results", "$_self.cast<FunctionType>().getResults()">]> {
  let summary = "indirect call";
  let description = [{
    The `pop.indirect_call` operation invokes the function pointer contained
    in an SSA value. This call operation is not involved in elaboration.

    Example:

    ```mlir
    kgen.func @call_this(%fn: (index) -> index) {
      %0 = index.constant 1
      %1 = pop.indirect_call %fn(%0) : (index) -> index
    }
    ```
  }];

  let arguments = (ins FunctionType:$callee, Variadic<AnyType>:$arguments);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee `(` $arguments `)` attr-dict `:` type($callee)";
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

def POP_StackAllocationOp : POPOp<"stack_allocation",
    [DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "scoped stack allocation operation";
  let description = [{
    The `pop.stack_allocation` operation allocates statically-sized scoped
    memory on the stack. The operation returns a pointer to the allocated memory
    of count elements aligned to the element type. An optional `align` attribute
    can be provided to specify the alignment of the allocation.

    Example:

    ```mlir
    %0 = pop.stack_allocation count x !pop.simd<4, f32>
    %1 = pop.stack_allocation count x !pop.scalar<ui64>
    %1 = pop.stack_allocation count x !pop.scalar<ui64> align 8
    ```

    The allocated memory is automatically released when control tranfers back
    from the region of its closest surrounding operation.

    Example:

    ```mlir
    kgen.generator @stack_memory<count, type: type>() {
      // This memory is allocated at the start of the function call and released
      // at the end.
      %0 = pop.stack_allocation count x type

      scf.if %cond {
        // This memory is allocated when control flow branches into the 'then'
        // region and released when control flow transfer back.
        %1 = pop.stack_allocation count x type
      }

      scf.for %iv = %lb to %ub step %step {
        // This memory is allocated each time the loop iterates and released
        // at the end of each iteration.
        %2 = pop.stack_allocation count x type
      }
    }
    ```

    If stack memory escapes the scope of its enclosing region, interacting with
    the memory outside its lifetime is undefined behaviour.

    Example:

    ```mlir
    // Scoped to the enclosing region.
    %0 = pop.stack_allocation count x type

    %mem = scf.if %cond -> !pop.pointer<type> {
      // OK: The memory is scoped to the enclosing region.
      scf.yield %0 : !pop.pointer<type>
    } else {
      // Scoped to the 'else' region.
      %1 = pop.stack_allocation count x type

      // The memory is escaping its enclosed region.
      scf.yield %1 : !pop.pointer<type>
    }

    // NOT OK: Loading pointer to invalid memory.
    %v = pop.load %mem : !pop.pointer<type>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$count,
                       OptionalAttr<TypedAttrInterface>:$alignment);
  let results = (outs Res<POP_PointerType, "", [MemAlloc]>:$result);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    custom<IndexParamValue>($count) `x` custom<PointerOf>(type($result))
    custom<OptionalAlignmentParamValue>($alignment) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$result, "TypedAttr":$count)>,
    OpBuilder<(ins "Type":$result, "int64_t":$count)>
  ];
}

//===----------------------------------------------------------------------===//
// MemcpyOp
//===----------------------------------------------------------------------===//

def POP_MemcpyOp : POPOp<"memcpy", [
    CopyOpInterface,
    AllTypesMatch<["dest", "src"]>,]> {
  let summary = "copies memory from one location to another";
  let description = [{
    The `pop.memcpy` operation copies memory from the src operand to the
    destination operand. The element type of both the source and destination
    locations must be the same. The the source location to the destination
    location must either be equal or non-overlapping. It copies `size` bytes of
    memory over. The `pop.memcpy` does not return a value and lowers to either
    an `"llvm.intrin.memcpy"` or `"llvm.intr.memcpy.inline"` LLVM dialect Op.

    The first argument is a pointer to the destination, the second is a pointer
    to the source. The third argument is an integer argument specifying the
    number of bytes to copy.

    If `size` is 0, it is no-op modulo the behavior of attributes attached to
    the arguments. If `size` is not a well-defined value, the behavior is
    undefined. If `size` is not zero, both `dest` and `src` should be
    well-defined, otherwise the behavior is undefined.

    The `pop.memcpy` takes optional modifiers to either inline or mark the
    memcpy as a volatile access. Those are specified via the `inline` and
    `volatile` tags respectively.

    Example:

    ```mlir
    // Copy size bytes from src to dest.
    pop.memcpy %dest, %src, %size : !pop.pointer<scalar<type>>

    // Generate an inlined memcpy which copies size bytes from src to dest.
    pop.memcpy inline %dest, %src, %size : !pop.pointer<scalar<i32>>

    // Generate an inlined memcpy which copies size bytes from src to dest and
    // mark the memcpy as a volatile access.
    pop.memcpy inline volatile %dest, %src, %size : !pop.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins Arg<POP_PointerType, "", [MemWrite]>:$dest,
                       Arg<POP_PointerType, "", [MemRead]>:$src,
                       Index:$size,
                       OptionalAttr<UnitAttr>:$isInlined,
                       OptionalAttr<UnitAttr>:$isVolatile);

  let extraClassDeclaration = [{
    Value getSource() { return getSrc(); }
    Value getTarget() { return getDest(); }
  }];

  let assemblyFormat = [{
    (`inline` $isInlined^)? (`volatile` $isVolatile^)? operands attr-dict `:`
    qualified(type($src))
  }];
}

//===----------------------------------------------------------------------===//
// MemsetOp
//===----------------------------------------------------------------------===//

def POP_MemsetOp : POPOp<"memset"> {
  let summary = "fills memory with a given value";
  let description = [{
    The `pop.memset` op will fill a block of memory with a particular byte
    value. Note that, unlike the standard libc function, the `pop.memset` op
    does not return a value.

    The first argument is a pointer to the destination to fill, the second is a
    byte value with which to fill it, the third argument is an integer argument
    specifying the number of bytes to fill.

    If the size is 0, this is no-op modulo the behavior of attributes attached
    to the arguments. If size is not a well-defined value, the behavior is
    undefined. If size is not zero, dest should be well-defined, otherwise the
    behavior is undefined.

    The op takes an optional volatile attribute. If the volatile attribute is
    set, the `pop.memset` makes this a volatile operation. The detailed access
    behavior is not very cleanly specified and it is unwise to depend on it.

    Example:

    ```mlir
    // Memsets size bytes to %zero.
    pop.memset %dest, %zero, %size : !pop.pointer<scalar<type>>

    // Generate a memset and mark the memset as a volatile access.
    pop.memset volatile %dest, %zero, %size : !pop.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins Arg<POP_PointerType, "", [MemWrite]>:$dest,
                       UI8Scalar:$value, Index:$size,
                       OptionalAttr<UnitAttr>:$isVolatile);

  let assemblyFormat = [{
    (`volatile` $isVolatile^)? operands attr-dict `:` qualified(type($dest))
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
    %0 = pop.external_call @sqrt(%arg) : (!pop.scalar<f64>) -> !pop.scalar<f64>
    ```

    Variadic functions are those with trailing function arguments. A variadic
    external function is indicated by specifying an explicit signature of the
    non-variadic arguments.

    ```mlir
    pop.external_call @printf(%fmt, %v0, %v1) (!pop.pointer<scalar<si8>>) -> ()
      : (!pop.pointer<scalar<si8>>, !pop.scalar<f32>, !pop.scalar<ui32>) -> ()
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
// GlobalConstantOp
//===----------------------------------------------------------------------===//

def POP_GlobalConstantOp : POPOp<"global_constant",
    [Pure, DeclareOpInterfaceMethods<ElaboratorOpInterface,
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
    %0 = pop.global_constant(5 : i32) : !pop.scalar<ui32>
    %1 = pop.global_constant(12.0) : type
    %2 = pop.global_constant(#M.dense_array<4, 2>) : !pop.simd<2, dtype>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs POP_PointerType:$result);

  let assemblyFormat = [{
    `(` $value `)` attr-dict `:` custom<PointerOf>(type($result))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndexToPointerOp
//===----------------------------------------------------------------------===//

def POP_IndexToPointerOp : POPOp<"index_to_pointer", [
    Pure,
    DeclareOpInterfaceMethods<CastOpInterface>
  ]> {
  let summary = "converts an index value to a pointer type.";
  let description = [{
    The `pop.index_to_pointer` instruction takes a simd vector of index dtype
    to cast, and a type to cast it to, which must be a pointer type or a simd
    of address dtype.

    Example:

    ```mlir
    %addr0 = pop.index_to_pointer %idx : index to !pop.pointer<scalar<f32>>
    %addr1 = pop.index_to_pointer %idx : index to !pop.pointer<simd<4, type>>
    %addr2 = pop.index_to_pointer %idx : index to !pop.scalar<address>
    %addr4 = pop.index_to_pointer %idx : !pop.simd<4, index> to
                                         !pop.simd<4, address>

    ```
  }];

  let arguments = (ins UnboundOrIndexSIMDOrIndex:$value);
  let results = (outs UnboundOrPointerOrSIMDAddressType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` qualified(type($value)) `to` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

def POP_PointerToIndexOp : POPOp<"pointer_to_index", [
    Pure,
    DeclareOpInterfaceMethods<CastOpInterface>
  ]> {
  let summary = "gets the address of the pointer as an index type.";
  let description = [{
    The `pop.pointer_to_index` instruction takes a pointer or a simd value of
    address dtype to cast and gets its address as a simd of index dtype.

    Example:

    ```mlir
    %addr0 = pop.pointer_to_index %ptr : !pop.pointer<scalar<f32>> to index
    %addr1 = pop.pointer_to_index %idx : !pop.pointer<simd<4, type>> to index
    %addr2 = pop.pointer_to_index %idx : !pop.scalar<address> to index
    %addr4 = pop.pointer_to_index %idx : !pop.simd<4, address> to
                                         !pop.simd<4, index>
    ```
  }];

  let arguments = (ins UnboundOrPointerOrSIMDAddressType:$value);
  let results = (outs UnboundOrIndexSIMDOrIndex:$result);
  let assemblyFormat = [{
    $value attr-dict `:` qualified(type($value)) `to` qualified(type($result))
  }];
}

//===----------------------------------------------------------------------===//
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

def POP_CastToBuiltinOp : POPOp<"cast_to_builtin", [Pure]> {
  let summary = "Unmaterialized type lowering conversion.";
  let description = [{
    The `pop.cast_to_builtin` operation casts a POP type to an MLIR builtin
    type. If the types have the same underlying implementation, then this
    operation is a no-op adaptor between two type systems. It can, for example,
    cast a `!pop.scalar` to an MLIR builtin integer or float type, or a
    `pop.simd` to a `vector.

    Example:

    ```mlir
    kgen.func @llvm_ops(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
      %0 = pop.cast_to_builtin %arg0 : !pop.scalar<f32> to f32
      %1 = llvm.fadd %0, %0 : f32
      %2 = pop.cast_from_builtin %1 : f32 to !pop.scalar<f32>
      kgen.return %2 : !pop.scalar<f32>
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
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

def POP_CastFromBuiltinOp : POPOp<"cast_from_builtin", [Pure]> {
  let summary = "Unmaterialized type raising conversion.";
  let description = [{
    The `pop.cast_from_builtin` operation casts an MLIR builtin type to a POP
    type. If the types have the same underlying implementation, then this
    operation is a no-op adaptor between two type systems. It can, for example,
    cast an MLIR builtin integer or float type to a `!pop.scalar`, or a 1-D
    vector to `!pop.simd`.

    Example:

    ```mlir
    kgen.func @llvm_ops(%arg0: !pop.scalar<f32>) -> !pop.scalar<f32> {
      %0 = pop.cast_to_builtin %arg0 : !pop.scalar<f32> to f32
      %1 = llvm.fadd %0, %0 : f32
      %2 = pop.cast_from_builtin %1 : f32 to !pop.scalar<f32>
      kgen.return %2 : !pop.scalar<f32>
    }
    ```

    Importantly, this operation can adapt parametric constants for use in
    parametric operations.

    Example:

    ```mlir
    kgen.generator @addN<N: si32>(%arg0: !pop.scalar<si32>)
        -> !pop.scalar<si32> {
      %N = kgen.param.constant: si32 = <N>
      %rhs = pop.cast_from_builtin %N : si32 to !pop.scalar<si32>
      %result = pop.add %arg0, %rhs : !pop.scalar<f32>
      kgen.return %result : !pop.scalar<f32>
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
// CallLLVMIntrinsicOp
//===----------------------------------------------------------------------===//

def POP_CallLLVMIntrinsicOp : POPOp<"call_llvm_intrinsic", [Pure]> {
  let summary = "Call to an LLVM intrinsic function.";
  let description = [{
    The `pop.call_llvm_intrinsic` operation calls an llvm intrinsic using the
    `llvm.call_intrinsic` op.

    Example:

    ```mlir
      %0 = pop.call_llvm_intrinsic "llvm.round"(%arg0) :
        !pop.scalar<f32> -> !pop.scalar<f32>
    ```
  }];
  let arguments = (ins StrAttr:$intrin, Variadic<AnyType>:$args);
  let results = (outs Optional<AnyType>:$results);
  let assemblyFormat = [{
    $intrin `(` $args `)` attr-dict `:`functional-type($args, $results)
  }];
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

def POP_InlineAsmOp : POPOp<"inline_asm"> {
  let description = [{
    The `pop.inline_asm` mirrors the underlying LLVM semantics with a notable
    exception: the embedded `asm_string` is not allowed to define or reference
    any symbol or any global variable: only the operands of the op may be read,
    written, or referenced.
    Attempting to define or reference any symbol or any global behavior is
    considered undefined behavior at this time.
  }];
  let arguments = (
    ins Variadic<AnyType>:$operands,
        StrAttr:$assembly,
        StrAttr:$constraints,
        UnitAttr:$hasSideEffects,
        UnitAttr:$isStackAligned,
        OptionalAttr<ArrayAttr>:$operandAttrs);

  let results = (outs Optional<AnyType>:$res);

  let assemblyFormat = [{
    (`side_effecting` $hasSideEffects^)?
    (`stack_aligned` $isStackAligned^)?
    (`operand_attrs` `=` $operandAttrs^)?
    attr-dict
    $assembly `,` $constraints
    operands `:` functional-type(operands, results)
   }];
}

//===----------------------------------------------------------------------===//
// PartialApplyOp
//===----------------------------------------------------------------------===//

def POP_PartialApplyOp : POPOp<"partial_apply",
      [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "partially apply arguments to a callable";
  let description = [{
    The `pop.partial_apply` operation takes a callable function
    type or !pop.closure type and binds values to its arguments, partially
    specializing the callable and returning a !pop.closure. The operation can
    bind any number of the callee's arguments.

    Example:

    ```mlir
    // The result has the closure type
    // `!pop.closure<(index, f32) -> !pop.scalar<address>>`.
    %0 = pop.partial_apply %fn(?, %arg0, ?, %arg1)
      : (index, i32, f32, !pop.scalar<index>) -> !pop.scalar<address>
    // Using partial_apply on a closure.
    %1 = pop.partial_apply %closure(?, %arg0, ?, %arg1)
      : !pop.closure<(index, i32, f32, <pop.scalar<index>) -> !pop.scalar<address>>

    ```

    The syntax of the operation uses `?` to denote arguments that are left
    unbound. These arguments form the signature of the resulting callable.
    In the above example, the second and fourth arguments to the callable `%fn`
    are bound to `%arg0` and `%arg1` respectively, returning a callable that
    has two fewer arguments.
  }];

  let arguments = (ins AnyTypeOf<[FunctionType, POP_ClosureType]>:$callee,
                       Variadic<AnyType>:$inputs,
                       DenseI64ArrayAttr:$boundInputs);
  let results = (outs POP_ClosureType:$result);
  let assemblyFormat = [{
    $callee `` custom<BoundInputs>($inputs, $boundInputs, type($inputs),
                                   type($result), type($callee)) attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}


//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

def POP_CallIndirectOp : POPOp<"call_indirect"> {
  let summary = "call a signature-typed value";
  let description = [{
    The `pop.call_indirect` operation takes an SSA value of `!pop.closure` type
    or function type and invokes it with the provided operands as arguments.

    Example:

    ```mlir
    %0 = pop.call_indirect %fn(%arg0, %arg1)
      : (index, !kgen.declref<@Foo>) -> (!pop.simd<2, si32>, !kgen.declref<@Bar>)
    //
    %0 = pop.call_indirect %closure(%arg0, %arg1)
      : !pop.closure<(index, !kgen.declref<@Foo>) -> (!pop.simd<2, si32>, !kgen.declref<@Bar>)>
    ```
  }];

  let arguments = (ins AnyTypeOf<[FunctionType, POP_ClosureType]>:$callee,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict
    `` custom<CallIndirectCalleeAndInputResultTypes>(type($inputs), type($results),
                                            type($callee))
  }];

  let hasCanonicalizeMethod = 1;
}
