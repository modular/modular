//===- POPOps.td ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the POP dialect.
//
//===----------------------------------------------------------------------===//

include "KGEN/MetaDialect/MetaDialect.td"
include "KGEN/MetaDialect/MetaTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def ConstantOp : POPOp<"constant", [
      ConstantLike,
      NoSideEffect,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "integer or floating point constant";
  let description = [{
    The `constant` operation produces a meta.scalar type equal to the
    input integer or floating-point constant specified. The
    input value need not match the dtype of the output meta.scalar but
    must be coerscible to it.

    Example:

    ```mlir
    // ok because the input type and the meta.scalar dtype are the same.
    %0 = pop.constant(42 : si32) : !meta.scalar<si32>
    %1 = pop.constant(42.0 : f64) : !meta.scalar<f64>
    // ok because 42:i64 can be represented as an f32
    %1 = pop.constant(42) : !meta.scalar<f32>
    // ok because 42:i64 can be represented as an f64
    %1 = pop.constant(42) : !meta.scalar<f64>
    // ok because 0.021:f64 can be represented as an f32
    %1 = pop.constant(0.021) : !meta.scalar<f32>
    // not ok, because 16777217:i32 cannot be represented as an f32
    %1 = pop.constant(16777217) : !meta.scalar<f32>
    ```
  }];
  let arguments = (ins AnyAttr:$value);
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

def AbsOp : POP_MetaArithmeticUnaryOp<"abs"> {
  let summary = "absolute value";
  let description = [{
    The `abs` operation takes a meta.scalar or meta.simd type and
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

def NegOp : POP_MetaArithmeticUnaryOp<"neg"> {
  let summary = "negation";
  let description = [{
    The `neg` operation takes a meta.scalar or meta.simd type and
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

def AddOp : POP_MetaArithmeticBinaryOp<"add", [Commutative]> {
  let summary = "addition";
  let description = [{
    The `add` operation takes two meta.scalar or meta.simd types
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

def SubOp : POP_MetaArithmeticBinaryOp<"sub"> {
  let summary = "subtraction";
  let description = [{
    The `sub` operation takes two meta.scalar or meta.simd types
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

def MulOp : POP_MetaArithmeticBinaryOp<"mul", [Commutative]> {
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
// CopySignOp
//===----------------------------------------------------------------------===//

def CopySignOp : POP_MetaArithmeticBinaryOp<"copysign"> {
  let summary = "copy sign";
  let description = [{
    The `copysign` operation takes two meta.scalar or meta.simd types
    (a magnitude `a` and a sign `b`) of floating-point dtype. The
    `copysign` operation copies the sign from `b` to the `a`. The
    operand and result types must be the same and must have a
    floating-point dtype.

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

def FMAOp : POP_MetaArithmeticTernaryOp<"fma"> {
  let summary = "fused-multiply add";
  let description = [{
    The `fma` operation takes three meta.scalar or meta.simd types
    (`a`, `b`, and `c`) and computes the fma operation `a * b + c`.
    The operand and result types must be the same.

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

def CmpOp : POPOp<"cmp", [NoSideEffect, SameTypeOperands,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "comparison";
  let description = [{
    The `cmp` operation takes a comparison predicate and two meta.scalar or
    meta.simd types (`lhs` and `rhs`) and computes the resulting comparison
    elementwise.

    The comparison semantics between floating point dtypes is the same as in C.

    Example:

    ```mlir
    %b0 = pop.cmp lt, %a, %b : !meta.scalar<type>
    %b1 = pop.cmp ge, %c, %d : !meta.simd<?, type>
    ```
  }];

  let arguments = (ins CmpPredicateAttr:$pred, ScalarOrSIMD:$lhs,
                       ScalarOrSIMD:$rhs);
  let results = (outs BoolScalarOrSIMD:$result);

  let assemblyFormat = "$pred `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}


//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : POPOp<"select",
        [NoSideEffect,
         AllTypesMatch<["trueValue", "falseValue", "result"]>,
         TypesMatchWith<"bool of same branch type", "result", "condition",
                        "getBoolOfSameParentType($_self)">
        ]> {
  let summary = "select";
  let description = [{
    The `select` operation takes a condition predicate and two meta.scalar or
    meta.simd types (`then` and `else`) and chooses one value based on a
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
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : POPOp<"cast", [NoSideEffect]> {
  let summary = [{
    Perform an element-wise cast between SIMD vectors types or between scalar
    types. SIMD vectors must have the same size. The cast semantics are the same
    as C casts. Only bool, integer, and floating point dtypes are supported.

    Example:

    ```mlir
    %0 = meta.cast %a : !meta.scalar<si32> to !meta.scalar<f32>
    %1 = meta.cast %b : !meta.scalar<ui64> to !meta.scalar<si32>
    %2 = meta.cast %c : !meta.simd<4, f64> to !meta.simd<4, ui32>
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

def SIMDExtractElementOp : POPOp<"simd.extractelement", [NoSideEffect,
     ScalarOfSameDType<"vector", "result">]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a SIMD vector and an index position and extracts the meta.scalar at
    that position. The position must be of type index.

    Example:

    ```mlir
    %1 = pop.simd.extractelement %0[%position]: !meta.simd<16, f32>
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

def SIMDInsertElementOp : POPOp<"simd.insertelement", [NoSideEffect,
      ScalarOfSameDType<"vector", "value">, AllTypesMatch<["result", "vector"]>]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a source value, SIMD vector, and an index position and inserts the
    source value into the vector at the proper position. The position must be
    of type index.

    Example:

    ```mlir
    %1 = pop.simd.extractelement %val, %vec[%position]: !meta.simd<16, f32>
    ```
  }];

  let arguments = (ins SIMDType:$vector, ScalarType:$value, Index:$position);
  let results = (outs SIMDType:$result);

  let assemblyFormat = [{
    $value `,` $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

def SIMDSplatOp : POPOp<"simd.splat", [NoSideEffect,
      ScalarOfSameDType<"result", "scalar">]> {
  let summary = "SIMD splat";
  let description = [{
    The `simd.splat` operation takes a scalar value and replicates it across
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
// BufferLoadOp
//===----------------------------------------------------------------------===//

def BufferLoadOp : POPOp<"buffer.load", [
     ScalarOfSameDType<"buffer", "result">]> {
  let summary = "buffer load operation";
  let description = [{
    Takes a Buffer and an index position and extracts the meta.scalar at
    that position. The position must be of type index.

    Example:

    ```mlir
    %1 = pop.buffer.load %0[%position]: !meta.buffer<size, dtype>
    ```
  }];

  let arguments = (ins Arg<BufferType, "", [MemRead]>:$buffer, Index:$position);
  let results = (outs ScalarType:$result);

  let assemblyFormat = [{
    $buffer `[` $position `]` attr-dict `:` qualified(type($buffer))
  }];
}

//===----------------------------------------------------------------------===//
// BufferStoreOp
//===----------------------------------------------------------------------===//

def BufferStoreOp : POPOp<"buffer.store", [
     ScalarOfSameDType<"buffer", "value">]> {
  let summary = "buffer store operation";
  let description = [{
    Takes a source value, buffer, and an index position and inserts the
    source value into the buffer at the proper position. The position must be
    of type index.

    Example:

    ```mlir
    pop.buffer.store %val, %buf[%position]: !meta.buffer<size, dtype>
    ```
  }];

  let arguments = (ins Arg<BufferType, "", [MemWrite]>:$buffer,
                       ScalarType:$value, Index:$position);

  let assemblyFormat = [{
    $value `,` $buffer `[` $position `]` attr-dict `:` qualified(type($buffer))
  }];
}
