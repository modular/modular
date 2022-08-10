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
  let results = (outs ScalarType:$result);

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
                      DeclareOpInterfaceMethods<InferTypeOpInterface,
                                                ["inferReturnTypes"]>]>,
    Arguments<(ins AnyTypeOf<[ScalarType, SIMDType]>:$operand)>,
    Results<(outs AnyTypeOf<[ScalarType, SIMDType]>:$result)>;


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
                      DeclareOpInterfaceMethods<InferTypeOpInterface,
                                                ["inferReturnTypes"]>]>,
    Arguments<(ins AnyTypeOf<[ScalarType, SIMDType]>:$lhs,
                   AnyTypeOf<[ScalarType, SIMDType]>:$rhs)>,
    Results<(outs AnyTypeOf<[ScalarType, SIMDType]>:$result)>;


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
// CopySign
//===----------------------------------------------------------------------===//

def CopySignOp : POP_MetaArithmeticBinaryOp<"copysign"> {
  let summary = "copy sign";
  let description = [{
    The `copysign` operation takes two meta.scalar or meta.simd types
    (a magntitude `a` and a sign `b`) of floating-point dtyle. The
    `copysign` operation copies the sign from `b` to the `a`. The
    operand and result types must be the same and must have a
    floating-point dtype.

    Example:

    ```mlir
    %c = pop.copysign %a, %b : !meta.scalar<type>
    %c = pop.copysign %a, %b : !meta.simd<size, type>
    ```
  }];
  let arguments = (ins AnyTypeOf<[UnboundOrFloatingPointScalarType, UnboundOrFloatingPointSIMDType]>:$lhs,
                       AnyTypeOf<[UnboundOrFloatingPointScalarType, UnboundOrFloatingPointSIMDType]>:$rhs);
}

//===----------------------------------------------------------------------===//
// TernaryOps
//===----------------------------------------------------------------------===//

// Base class for ternary arithmetic pop operations.
class POP_ArithmeticTernaryOp<string mnemonic, list<Trait> traits = []> :
       POPOp<mnemonic, traits # [NoSideEffect, SameOperandsAndResultType]> {
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($result)";
}

// Base class for ternary operations.
class POP_MetaArithmeticTernaryOp<string mnemonic, list<Trait> traits = []> :
    POP_ArithmeticTernaryOp<mnemonic, traits # [
                      DeclareOpInterfaceMethods<InferTypeOpInterface,
                                                ["inferReturnTypes"]>]>,
    Arguments<(ins AnyTypeOf<[ScalarType, SIMDType]>:$a,
                   AnyTypeOf<[ScalarType, SIMDType]>:$b,
                   AnyTypeOf<[ScalarType, SIMDType]>:$c)>,
    Results<(outs AnyTypeOf<[ScalarType, SIMDType]>:$result)>;

//===----------------------------------------------------------------------===//
// FMAOp
//===----------------------------------------------------------------------===//

def FMAOp : POP_MetaArithmeticTernaryOp<"fma"> {
  let summary = "fused-multiply add";
  let description = [{
    The `fma` operation takes three meta.scalar or meta.vector types
    (`a`, `b`, and `c`) and computes the fma operation `a * b + c`.
    The operand and result types must be the same.

    Example:

    ```mlir
    %d = pop.fma %a, %b, %c : !meta.scalar<type>
    %d = pop.fma %a, %b, %c : !meta.simd<size, type>
    ```
  }];
}
