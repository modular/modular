//===- KGENOps.td ---------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This defines operations in the KGEN dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "KGEN/KGENDialect/KGENDeclInterface.td"

// Base class for the operation in this dialect.
class KGENOp<string mnemonic, list<Trait> traits = []> :
    Op<KGENDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp
  : KGENOp<"func", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                    Symbol, IsolatedFromAbove,
                    SingleBlockImplicitTerminator<"ReturnOp">,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                    DeclareOpInterfaceMethods<KGENDeclInterface, ["getConstraints"]>]> {
  let summary = "concrete function definition";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$sym_visibility,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getBody().front();
    }

    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Create a func with a region but no body block.  The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name, "StringAttr":$visibility,
                   "FunctionType":$signature,
                   "ArrayRef<Type>":$outputParamTypes)>,
    // Create a func with an empty body, `argLocs` specifies the locations for
    // all the block arguments.
    OpBuilder<(ins "StringAttr":$name, "StringAttr":$visibility,
                   "FunctionType":$signature,
                   "ArrayRef<Type>":$outputParamTypes,
                   "ArrayRef<Location>":$argLocs)>,
  ];
}

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

def GeneratorOp
  : KGENOp<"generator", [HasParent<"mlir::ModuleOp">, FunctionOpInterface,
                         Symbol, IsolatedFromAbove,
                         SingleBlockImplicitTerminator<"ReturnOp">,
                         DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                         DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "Function generator definition";

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$sym_visibility,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<FlatSymbolRefAttr>:$implements);
  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    using FunctionOpInterfaceTrait::getBody;
    using FunctionOpInterfaceTrait::front;
    Block *getBodyBlock() {
      return &getBody().front();
    }

    ReturnOp getReturnOp();

    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

def GeneratorInterfaceOp
  : KGENOp<"generator.interface",
           [HasParent<"mlir::ModuleOp">, FunctionOpInterface, Symbol,
            IsolatedFromAbove, DeclareOpInterfaceMethods<KGENDeclInterface>,
            DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Generator interface definition";

  /// input and result parameters are stored in a single flat array named
  /// `paramDecls` and the `numInputParameters` field indicates which are input
  /// and which are result parameters.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$sym_visibility,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       KTypeArrayAttr:$resultParamTypes,
                       ConstraintArrayAttr:$constraints,
                       OptionalAttr<EvalConfigurationArrayAttr>:$evalConfigs);

  // We want to work with the FunctionOpInterface, which always requires a body
  // region.  We are "declaration like" though, so ours never has a block.
  let regions = (region SizedRegion<0>:$body);

  let extraClassDeclaration = [{
    /// Returns the argument types of this invokable.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this invokable.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp
  : KGENOp<"call", [CallOpInterface,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                    IsolatedFromAbove]> {

  let summary = "Call a func, generator, or generator interface";

  let arguments =
   (ins FlatSymbolRefAttr:$callee, ParamBindArrayAttr:$paramValues,
        ParamDeclArrayAttr:$paramDecls, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let extraClassDeclaration = [{
    FunctionType getFunctionType() {
      return FunctionType::get(getContext(), getOperandTypes(), 
                               getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Return a signature for this call.
    SignatureType getSignature();
  }];

  let assemblyFormat = [{
    $callee `` custom<CallOpParams>($paramValues, $paramDecls) `(` $operands `)`
    attr-dict `:` functional-type($operands, results)
    custom<CallRegions>($regions, ref($paramValues))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "StringAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   "OperandRange":$operands)>,
  ];
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

// NOTE: This cannot be a CallOpInterface, because the callee is not a
// SymbolRefAttr, it is a different kind of attribute.
def CallParamOp : KGENOp<"call_param", [IsolatedFromAbove]> {
  let summary = "Call declaration specified by a parameter experession";

  let arguments =
   (ins ParameterExprAttr:$callee, ParamBindArrayAttr:$paramValues,
        ParamDeclArrayAttr:$paramDecls, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$regions);

  let extraClassDeclaration = [{
    FunctionType getFunctionType() {
      return FunctionType::get(getContext(), getOperandTypes(), 
                               getResultTypes());
    }

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return a signature for this call.
    SignatureType getSignature();
  }];

  let assemblyFormat = [{
    `` custom<CallParamCallee>($callee, $paramValues, $paramDecls, 
                            type($operands), type($results))
    `(` $operands `)`
    attr-dict
    custom<CallRegions>($regions, ref($paramValues))
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "TypedAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   "OperandRange":$operands)>,
  ];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

def ParamConstantOp : KGENOp<"param.constant",
                             [FirstAttrDerivedResultType, NoSideEffect,
                              ConstantLike]> {
  let summary = [{
    Return the value of a parameter expression as an SSA value that may be used
    by other ops.
  }];

  let arguments = (ins ParameterExprAttr:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat =
     "custom<ParamConstantOpValue>($value, qualified(type($result))) attr-dict";

  let builders = [OpBuilder<(ins "TypedAttr":$value)>];

  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def ParamDeclareOp : KGENOp<"param.declare", []> {
  let summary = [{
    Declare a new parameter with a value.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls,
                       ParameterExprAttr:$value);
  let assemblyFormat =
     "custom<ParamDeclareOpValue>($paramDecls, $value) attr-dict";

  let builders = [
    OpBuilder<(ins "ParamDeclAttr":$decl, "Attribute":$value)>,
  ];

  let extraClassDeclaration = [{
    /// Return the single parameter declaration as a well typed attribute.
    ParamDeclAttr getParamDecl();
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : KGENOp<"return", [Terminator, ReturnLike]> {
  let summary = "return operation for kgen declarations";
  let description = [{
    The `kgen.return` operation specifies the returned values and output
    parameters for a `kgen.func` or generator.

    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the
    `kgen.generator` (or enclosing region) that contains the operation. For
    example:

    ```mlir
      kgen.generator @foo() : i32, f8 {
        ...
        kgen.return %0, %1 : i32, f8
      }
    ```
  }];

  let arguments = (ins ParameterExprArrayAttr:$parameters,
                       Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    `` custom<ParameterValues>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let extraClassDeclaration = [{
    /// Verify that the operands of this ReturnOp match the specified set of
    /// types.
    LogicalResult checkArgumentTypes(ArrayRef<Type> resultParamTypes,
                                     TypeRange types);
  }];
}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

def ParamAssertOp : KGENOp<"param.assert", []> {
  let summary = "check a parameter expression at elaboration time";
  let description = [{
    The `kgen.param.assert` operation ensures that a parameter expression is
    true at elaboration time.  It is a more flexible form of the constraint list
    on a generator.
  }];

  let arguments = (ins I1ParameterExprAttr:$cond, StrAttr:$message);
  let assemblyFormat =
    "` ` `<` custom<ParamAssertOpValue>($cond) `>` `,` $message attr-dict";

  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

def IncludeOp : KGENOp<"include", []> {
  let summary = "include an external file";
  let description = [{
    The `kgen.include` operation includes an external file and makes all the
    symbols contained within available for use.

    This op only exists in the IR before the include is resolved, once it's
    resolved the IncludeOp is removed. Includes are resolved by the elaborator
    by providing a list of search paths.
  }];

  let arguments = (ins StrAttr:$fileName);
  let assemblyFormat = "$fileName attr-dict-with-keyword";
}

//===----------------------------------------------------------------------===//
// RebindOp
//===----------------------------------------------------------------------===//

def RebindOp : KGENOp<"rebind"> {
  let summary = "rebind parametric types";
  let description = [{
    The `kgen.rebind` operation rebinds values with parametric types into other
    parameter domains. The operation can be used to convert between parameter
    domains or between concrete types. During elaboration, rebind operations
    must resolve to the same input and output types. Otherwise, elaboration will
    fail.

    Example:

    ```mlir
    // Rebind a parameterized type to a concrete type.
    %0 = kgen.rebind %arg0 : !kgen.paramref<type> to !meta.scalar<f32>

    // Rebind between parameter domains.
    %1 = kgen.rebind %arg1 : !kgen.paramref<type> to !meta.simd<size, dtype>

    // Unbind a concrete type to one with a parameter.
    %2 = kgen.rebind %arg2 : !meta.scalar<f32> to !meta.scalar<dtype>

    // ERROR: Cannot rebind between different concrete types.
    %3 = kgen.rebind %arg2 : !meta.scalar<f32> to !meta.scalar<si32>
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// PrecompiledOpBase
//===----------------------------------------------------------------------===//

class PrecompiledOpBase<string mnemonic, dag additionalArgs>
  : KGENOp<"precompiled." # mnemonic, [HasParent<"mlir::ModuleOp">,
                                       FunctionOpInterface, Symbol]> {

  let arguments = !con((ins SymbolNameAttr:$sym_name,
                            StrAttr:$sym_visibility,
                            TypeAttrOf<FunctionType>:$function_type), additionalArgs);

  // We want to work with the FunctionOpInterface, which always requires a body
  // region.  We are "declaration like" though, so ours never has a block.
  let regions = (region SizedRegion<0>:$body);

  let extraClassDeclaration = [{
   /// Returns the argument types of this precompiled op.
   ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

   /// Returns the result types of this precompiled op.
   ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let extraClassDefinition = [{
    /// Parse an instance of this precompiled op.
    ParseResult $cppClass::parse(::mlir::OpAsmParser &parser,
                                 ::mlir::OperationState &result) {
      return parsePrecompiledOp(parser, result);
    }

    /// Print an instance of this precompiled op.
    void $cppClass::print(::mlir::OpAsmPrinter &p) {
      printPrecompiledOp(p, *this);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// PrecompiledFuncOp
//===----------------------------------------------------------------------===//

def PrecompiledFuncOp : PrecompiledOpBase<"func", (ins StrAttr:$func)> {
  let summary = "declare a precompiled `kgen.func`";
  let description = [{
    The `kgen.precompiled.func` operation provides an IR handle for a
    `kgen.func` that is already compiled/not possible to inline. It maintains
    all function attributes that were on the original function, but elides the
    body in favor of storing it as a cache key that can be used for lookups as
    needed.
  }];
}

//===----------------------------------------------------------------------===//
// PrecompiledLLVMOp
//===----------------------------------------------------------------------===//

def PrecompiledLLVMOp : PrecompiledOpBase<"llvm",
    (ins TargetInfoAttr:$compiledFor, StrAttr:$ir, StrAttr:$llvm)> {
  let summary = "declare a precompiled `kgen.func`'s LLVM lowering";
  let description = [{
    The `kgen.precompiled.llvm` operation provides an IR handle for a
    `kgen.func` that is already compiled to LLVM. It stores a cache key to
    the LLVM representation of this function (the `llvm` attr), as well as an
    optional back pointer to the `kgen.precompiled.func` op (the `ir` attr) for
    back-tracking up the compilation stack.

    `kgen.precompiled.llvm` is target-specific, as the LLVM lowering requires
    a data layout.
  }];
}

//===----------------------------------------------------------------------===//
// PrecompiledObjectOp
//===----------------------------------------------------------------------===//

def PrecompiledObjectOp : PrecompiledOpBase<"object",
    (ins TargetInfoAttr:$compiledFor, StrAttr:$llvm, StrAttr:$object)> {
  let summary = "declare a precompiled `kgen.func`'s cached object";
  let description = [{
    The `kgen.precompiled.object` operation provides an IR handle for a
    `kgen.func` that is already compiled to an object file. It stores a cache
    key to the object file (the `object` attr) as well as an optional back
    pointer to the `kgen.precompiled.llvm` op (the `llvm` attr) for
    back-tracking up the compilation stack.

    `kgen.precompiled.object` is target-specific, it contains the raw bytes
    of a machine-specific object file.
  }];
}
