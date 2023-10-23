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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ExprNodes.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoParser/StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {

using ClosureHash = std::pair<SignatureType, StringAttr>;

/// Top level types are the types of the Closure Wrapper function pointer
/// fields.
struct TopLevelTypes {
  Type callFuncFieldType;
  Type copyFuncFieldType;
  Type delFuncFieldType;
};

/// A ClosureCache stores previously generated closures to prevent duplicate
/// definitions from being generated.
class ClosureCache {
public:
  virtual ~ClosureCache() = default;
  virtual StructDeclOp getExisting(ClosureHash key) = 0;
  virtual void storeClosure(ClosureHash key, StructDeclOp closure) = 0;
};

class ClosureEmitter : public StructEmitter {
public:
  ClosureEmitter(ASTDecl &moduleDecl, SharedState &shared);

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp
  createClosureWrapperStructDecl(StringAttr name,
                                 LITSignatureType signatureType,
                                 SMLoc nestedFunctionOrTypeLocation);

  /// Generate a Closure Implementation Struct, a struct that contains the
  /// capture list.
  StructDeclOp replaceNestedFunctionWithClosureImplStructDecl(
      SMLoc loc, ASTDecl &nestedFunctionDecl,
      CaptureTraversableMap capturedParams, ClosureCache &cache);

  /// Generate an initializer on the ClosureWrapper that accepts a ClosureImpl
  /// instance.
  LIT::FuncOp createWrapperInitWithImpl(StructDeclOp closureWrapper,
                                        StructDeclOp closureImpl,
                                        SMLoc location);

  /// Generate a unique name for a closure class.
  static StringAttr getClosureNameFromType(StringRef prefix,
                                           FileModuleOp fileModuleOp,
                                           SignatureType signatureType);

private:
  MLIRContext *ctx;
  /// The decl of the surrounding module where code should be synthesized.
  ASTDecl &moduleDecl;
  /// A synthetic node to carry location information for emitting IR.
  SyntheticNode node;

  /// The surrounding file module operation.
  FileModuleOp fileModuleOp;

  // Cached attributes and types.
  StringAttr selfName;
  StringAttr otherName;
  StringAttr ptrToImplName;
  StringAttr dtorFieldAttr;
  StringAttr copyFieldAttr;
  StringAttr callFieldAttr;
  StringAttr callMethodAttr;
  PointerType opaquePtrType;

  /// Given a signature of a function, create a new signature by inserting a
  /// closure argument at index 0 or 1 depending on the result type.
  LITSignatureType
  addClosureSelfArgToFunctionSignature(Type closureType,
                                       LITSignatureType sig) const;
  /// Given a Closure struct and parameter values, create the specialized self
  /// type.
  Type makeClosureImplSelfType(StructDeclOp closureImpl,
                               SmallVector<ParamDeclAttr> paramValues);

  /// Given a closure wrapper, collect the top level function types.
  TopLevelTypes collectTopLevelFunctionTypes(StructDeclOp closureWrapper);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
