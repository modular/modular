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
                    DeclareOpInterfaceMethods<KGENDeclInterface, ["getConstraints"]>]> {
  let summary = "concrete function definition";

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$resultParamDecls);
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
  let hasRegionVerifier = 1;

  let builders = [
    // Create a kernel with a region but no body block.  The caller must create
    // it and fill it in.
    OpBuilder<(ins "StringAttr":$name, "FunctionType":$signature,
                   "ArrayRef<ParamDeclAttr>":$outputParams)>,
    // Create a kernel with an empty body, `argLocs` specifies the locations for
    // all the block arguments.
    OpBuilder<(ins "StringAttr":$name, "FunctionType":$signature,
                   "ArrayRef<ParamDeclAttr>":$outputParams,
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
  let summary = "Kernel generator definition";

  /// If the `implements` attribute is present then this generator is an
  /// implementation of the specified generator.interface.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       ParamDeclArrayAttr:$resultParamDecls,
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
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GeneratorInterfaceOp
//===----------------------------------------------------------------------===//

def GeneratorInterfaceOp
  : KGENOp<"generator.interface",
           [HasParent<"mlir::ModuleOp">, FunctionOpInterface, Symbol,
            IsolatedFromAbove, DeclareOpInterfaceMethods<KGENDeclInterface>]> {
  let summary = "Generator interface definition";

  /// input and result parameters are stored in a single flat array named
  /// `paramDecls` and the `numInputParameters` field indicates which are input
  /// and which are result parameters.
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ParamDeclArrayAttr:$paramDecls,
                       ParamDeclArrayAttr:$resultParamDecls,
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
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp
  : KGENOp<"call", [CallOpInterface,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "Call a kernel or generator";

  let arguments =
   (ins FlatSymbolRefAttr:$callee, ParamBindArrayAttr:$paramValues,
        ParamDeclArrayAttr:$paramDecls, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    FunctionType getCalleeType();

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
  }];

  let assemblyFormat = [{
    $callee `` custom<CallOpParams>($paramValues, $paramDecls) `(` $operands `)`
    attr-dict `:` functional-type($operands, results)
  }];

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "StringAttr":$callee,
                   "ArrayRef<ParamBindAttr>":$inputParams,
                   "ArrayRef<ParamDeclAttr>":$resultParams,
                   "OperandRange":$operands)>,
  ];
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
  let hasFolder = true;
}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

def ParamDeclareOp : KGENOp<"param.declare", [NoSideEffect]> {
  let summary = [{
    Declare a new parameter with a value.
  }];

  let arguments = (ins ParamDeclSize1ArrayAttr:$paramDecls,
                       ParameterExprAttr:$value);
  let results = (outs);
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
  let summary = "return operation for kgen generators";
  let description = [{
    The `kgen.return` operation specifies the returned values for a generated
    kernel and output parameters from the metaprogram.

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

  let arguments = (ins ParamBindArrayAttr:$parameters,
                       Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    custom<ParameterBindings>($parameters) attr-dict
    ($operands^ `:` type($operands))?
  }];

  let extraClassDeclaration = [{
    /// Verify that the operands of this ReturnOp match the specified set of
    /// types.
    LogicalResult checkArgumentTypes(ArrayRef<ParamDeclAttr> paramDecls,
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
