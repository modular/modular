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
      SMLoc loc, ASTDecl &nestedFnDecl,
      ArrayRef<ParamDeclRefAttr> paramCaptures, LITSignatureType wrapperSig);

  /// Generate an initializer on the ClosureWrapper that accepts a ClosureImpl
  /// instance. The 'fromImplToWrapperParameterIndexMap' allows the caller to
  /// specify which parameters of the ClosureWrapper should be bound to the
  /// ClosureImpl.
  LIT::FuncOp createWrapperInitWithImpl(
      StructDeclOp closureWrapper, StructDeclOp closureImpl,
      SmallDenseMap<unsigned, unsigned> fromImplToWrapperParameterIndexMap,
      SMLoc location);

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
  RefType opaqueRefType;

  /// Given a closure wrapper, collect the top level function types.
  TopLevelTypes collectTopLevelFunctionTypes(StructDeclOp closureWrapper);
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_CLOSUREEMITTER_H
