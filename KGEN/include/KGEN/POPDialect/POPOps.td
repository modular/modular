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
    The `constant` operation produces a meta.scalar value equal to some
    integer or floating-point constant specified by an attribute.

    Example:

    ```mlir
    %0 = pop.constant(42 : si32) : !meta.scalar<si32>
    %1 = pop.constant(42 : f64) : !meta.scalar<f64>
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
// BinaryOp
//===----------------------------------------------------------------------===//

// Base class for binary arithmetic pop operations.
class POP_ArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
       POPOp<mnemonic, traits # [NoSideEffect, SameOperandsAndResultType]> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// Base class for integer binary operations.
class POP_MetaArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
    POP_ArithmeticBinaryOp<mnemonic, traits # [
                      DeclareOpInterfaceMethods<InferTypeOpInterface,
                                                ["inferReturnTypes"]>]>,
    Arguments<(ins ScalarType:$lhs, ScalarType:$rhs)>,
    Results<(outs ScalarType:$result)>;


//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : POP_MetaArithmeticBinaryOp<"add", [Commutative]> {
  let summary = "addition";
  let description = [{
    The `add` operation takes two meta variables numbers and returns their sum.
    The operand and result types must be the same.

    Example:

    ```mlir
    %a = pop.add %b, %c : !meta.scalar<type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SubtractOp
//===----------------------------------------------------------------------===//

def SubOp : POP_MetaArithmeticBinaryOp<"sub"> {
  let summary = "subtraction";
  let description = [{
    The `sub` operation subtracts two meta variables numbers.
    The operand and result types must be the same.

    Example:

    ```mlir
    %a = pop.sub %b, %c : !meta.scalar<type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : POP_MetaArithmeticBinaryOp<"mul", [Commutative]> {
  let summary = "multiplication";
  let description = [{
    The `mul` operation takes two meta variables numbers and returns their product.
    The operand and result types must be the same.

    Example:

    ```mlir
    %a = pop.mul %b, %c : !meta.scalar<type>
    ```
  }];
}
