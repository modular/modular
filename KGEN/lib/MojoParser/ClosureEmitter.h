//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Closure Emission.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CLOSUREEMITTER_H
#define KGEN_MOJOPARSER_CLOSUREEMITTER_H

#include "ExprNodes.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/SharedState.h"
#include "StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {
class TypeCheckedFnSignature;
class TypeCheckedParamList;

/// Top level types are the types of the Closure Wrapper function pointer
/// fields.
struct TopLevelTypes {
  Type callFuncFieldType;
  Type copyFuncFieldType;
  Type delFuncFieldType;
};

struct FnOpAttributes {
  /// Signature with index references to reference parameters
  FnTypeGeneratorType signature;

  /// the number of compiler generated arguments.
  unsigned numberOfImplicitClosureArgs;

  /// FunctionType with named references to parameters
  FunctionType functionType;
};
class ClosureEmitter : public StructEmitter {
public:
  ClosureEmitter(ASTDecl &moduleDecl, SharedState &shared);

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp
  createClosureWrapperStructDecl(StringAttr name,
                                 FnTypeGeneratorType signatureType,
                                 SMLoc nestedFunctionOrTypeLocation);

  /// Generate a Parametric Closure Wrapper Struct, a struct that contains a
  /// parametric field. Both the field and the struct must conform to the
  /// associated closure trait characterized by the signature of the closure.
  std::pair<StructDeclOp, TraitDeclOp> createParametricClosureWrapperStructDecl(
      StringAttr name, FnTypeGeneratorType signatureType,
      SMLoc nestedFunctionOrTypeLocation, InlineLevel inlineLevel);

  /// Generate a Closure Implementation Struct, a struct that contains the
  /// capture list.
  StructDeclOp replaceNestedFunctionWithClosureImplStructDecl(
      ArrayRef<Capture> captures, ArrayRef<ParamDeclRefAttr> paramCaptures,
      ASTDecl &nestedfnDecl, FnTypeGeneratorType wrapperSigGen);

  /// Generate an initializer on the ClosureWrapper that accepts a ClosureImpl
  /// instance.
  FnOp createWrapperInitWithImpl(StructDeclOp closureWrapper,
                                 StructDeclOp closureImpl, SMLoc location);
  Value emitClosureOp(ASTDecl &nestedFnDecl, ArrayRef<Capture> captures,
                      StructDeclOp wrapper, TraitDeclOp trait,
                      Location location);
  /// Given a type checked signature, create a new signature such that for every
  /// closure typed parameter an implicit argument is added to the signature and
  /// function. Note that closure parameters are renamed and the original name
  /// is mapped to the new argument. This is because we can extract a type from
  /// a value but we cannot extract a value from a type.
  FnOpAttributes computeFunctionSignature(ASTDecl &decl, StringAttr baseName,
                                          TypeCheckedFnSignature &tcSignature);

private:
  MLIRContext *ctx;
  /// The decl of the surrounding module where code should be synthesized.
  ASTDecl &moduleDecl;
  /// A synthetic node to carry location information for emitting IR.
  SyntheticNode node;

  /// The surrounding file module operation.
  FileModuleOp fileModuleOp;

  // Cached attributes and types.
  StringAttr selfName, otherName, ptrToImplName, dtorFieldAttr;
  StringAttr copyFieldAttr, callFieldAttr, callMethodAttr;
  PointerType opaquePtrType;

  /// Given a closure wrapper, collect the top level function types.
  TopLevelTypes collectTopLevelFunctionTypes(StructDeclOp closureWrapper);

  /// Synthesize the constructor for a closure wrapper struct from a bare
  /// function pointer of the same function signature.
  void synthesizeWrapperFnPtrCtor(ASTDecl &decl, ASTType selfType,
                                  FnTypeGeneratorType sig);
  /// Given a name, a list of builtin parent traits (like "Movable" for
  /// example), a location, and a populate method, return a trait declaration
  /// that inherits from the parent and contains the methods added to the
  /// function list populated by the populate method.
  std::pair<TraitDeclOp, ASTDecl *>
  createTraitOp(StringAttr name, ArrayRef<StringRef> parents,
                SMLoc nestedFunctionOrTypeLocation,
                llvm::function_ref<void(
                    ASTDecl &traitDecl,
                    DenseSet<std::pair<StringAttr, StringAttr>> &functions)>
                    populateTrait);
  /// Given a name and a trait decl, generate a struct that conforms to the
  /// trait and has a single field that also conforms to that same trait. For
  /// example, if the trait is:
  /// trait MyTrait:
  ///    fn doSomething(self, x:Int):
  ///       ...
  /// This method will generate a struct:
  /// struct MyTraitWrapper[T: MyTraitWrapper](MyTraitWrapper):
  ///    var field: T
  ///    fn doSomething(self, x:Int):
  ///       self.field.doSomething(x)
  /// This is useful in the context of emitting closures because in the case of
  /// closures "T" is an abstract type (ClosureType). Wrapping the closure
  /// instance in a struct renders it eligible to be handled properly by our
  /// check-lifetimes pass.
  StructDeclOp createStructWrapper(StringRef baseName, ASTDecl &traitDecl,
                                   SMLoc location);

  // Collect metadata used to augment function and type signatures.
  struct ClosureSignatureInfo {
    /// Param metadata
    SmallVector<ParamDeclAttr> implicitOriginDecls;

    /// arg metadata
    SmallVector<PogMetadataAttr> argPogs;
    SmallVector<Type> argTypes;
    SmallVector<ArgConvention> argConventions;
    SmallVector<SMLoc> locations;

    // move-only
    ClosureSignatureInfo() = default;
    ClosureSignatureInfo(const ClosureSignatureInfo &) = delete;
    ClosureSignatureInfo &operator=(const ClosureSignatureInfo &) = delete;
    ClosureSignatureInfo(ClosureSignatureInfo &&) = default;
    ClosureSignatureInfo &operator=(ClosureSignatureInfo &&) = default;
  };

  ClosureSignatureInfo remapClosureParameters(ASTDecl &decl,
                                              TypeCheckedParamList &paramList);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
