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
include "KGEN/POPDialect/POPDialect.td"
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
    The `constant` operation produces a meta.scalar type equal to the
    input integer or floating-point constant specified. The
    input value need not match the dtype of the output meta.scalar but
    must be coerscible to it.

    Example:

    ```mlir
    // ok because the input type and the meta.scalar dtype are the same.
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

def POP_NegOp : POP_MetaArithmeticUnaryOp<"neg"> {
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

def POP_AddOp : POP_MetaArithmeticBinaryOp<"add", [Commutative]> {
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

def POP_SubOp : POP_MetaArithmeticBinaryOp<"sub"> {
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
// ShlOp
//===----------------------------------------------------------------------===//

def POP_ShlOp : POP_MetaArithmeticBinaryOp<"shl"> {
  let summary = "shift left";
  let description = [{
    The pop.shl instruction returns the first operand shifted to the left a
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

def POP_FMAOp : POP_MetaArithmeticTernaryOp<"fma"> {
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

def POP_CmpOp : POPOp<"cmp", [NoSideEffect, SameTypeOperands,
                          DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "comparison";
  let description = [{
    The `cmp` operation takes a comparison predicate and two meta.scalar or
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
// BitCastOp
//===----------------------------------------------------------------------===//

def POP_BitcastOp : POPOp<"bitcast", [NoSideEffect,
                                  DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = [{
    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width.

    Example:

    ```mlir
    %0 = meta.bitcast %a : !meta.scalar<si32> to !meta.scalar<f32>
    %1 = meta.bitcast %b : !meta.scalar<ui64> to !meta.scalar<si64>
    %2 = meta.bitcast %c : !meta.simd<4, f64> to !meta.simd<4, si64>
    %3 = meta.bitcast %d : !meta.simd<2, f64> to !meta.simd<4, f32>
    ```
  }];

  let arguments = (ins ScalarOrSIMD:$input);
  let results = (outs ScalarOrSIMD:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def POP_CastOp : POPOp<"cast", [NoSideEffect]> {
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
    The `simd.shuffle` operation constructs a permutation of two vectors of the
    same length and returns a vector of the same element type with the same
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
// LoadOp
//===----------------------------------------------------------------------===//

def POP_LoadOp : POPOp<"load", [ScalarOfSameDType<"ptr", "result">]> {
  let summary = "pointer load operation";
  let description = [{
    Loads the data at the given pointer. Returns a scalar of the same dtype. The
    pointer cannot have a `?` dtype.

    Example:

    ```mlir
    %0 = pop.load %ptr0 : !meta.pointer<f32>
    %1 = pop.load %ptr1 : !meta.pointer<type>
    ```
  }];

  let arguments = (ins Arg<ConcretePointer, "", [MemRead]>:$ptr);
  let results = (outs ScalarType:$result);
  let assemblyFormat = "$ptr attr-dict `:` type($ptr)";

  let builders = [OpBuilder<(ins "Value":$ptr)>];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def POP_StoreOp : POPOp<"store", [ScalarOfSameDType<"ptr", "arg">]> {
  let summary = "pointer store operation";
  let description = [{
    Store the given scalar data to the given pointer of the same dtype. The
    pointer cannot have a `?` dtype.

    Example:

    ```mlir
    pop.store %0, %ptr0 : !meta.pointer<f32>
    pop.store %1, %ptr1 : !meta.pointer<type>
    ```
  }];

  let arguments = (ins ScalarType:$arg,
                       Arg<ConcretePointer, "", [MemWrite]>:$ptr);
  let assemblyFormat = "$arg `,` $ptr attr-dict `:` type($ptr)";
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

def POP_OffsetOp : POPOp<"offset", [AllTypesMatch<["result", "ptr"]>]> {
  let summary = "pointer offset operation";
  let description = [{
    The `offset` operation takes a pointer and increments or decrements its
    position by a certain amount.

    Example:

    ```mlir
    %addr = pop.offset %base[%idx] : !meta.pointer<f32>
    ```
  }];

  let arguments = (ins PointerType:$ptr, Index:$index);
  let results = (outs PointerType:$result);
  let assemblyFormat = [{
    $ptr `[` $index `]` attr-dict `:` qualified(type($ptr))
  }];
}

//===----------------------------------------------------------------------===//
// BufferStackAllocationOp
//===----------------------------------------------------------------------===//

def POP_BufferStackAllocationOp : POPOp<"buffer.stack_allocation"> {
  let summary = "stack memory allocation operation";
  let description = [{
    The `pop.buffer.stack_allocation` operation allocates memory on the stack.
    The allocated memory is automatically released when control transfers back
    from the region of its closest surrounding operation with an
    AutomaticAllocationScope trait. The amount of memory allocated is specified
    by its buffer type.

    The resulting buffer type must be concrete in both the size and dtype.
    I.e. The buffer type cannot have `?` as either the size or dtype.

    Example:

    ```mlir
    %0 = pop.buffer.stack_allocation : !meta.buffer<4, f32>
    %1 = pop.buffer.stack_allocation : !meta.buffer<size, f32>
    %2 = pop.buffer.stack_allocation : !meta.buffer<size, type>
    // the following is not allowed
    %3 = pop.buffer.stack_allocation : !meta.buffer<?, f32>
    ```
  }];

  let results = (outs
    Res<ConcreteBuffer, "",
        [MemAlloc<AutomaticAllocationScopeResource>]>:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasVerifier = 1;
}
