//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Closure Emission.
//
//===----------------------------------------------------------------------===//

#ifndef CLOSUREEMITTER_H
#define CLOSUREEMITTER_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoParser/StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {

using ClosureHash = std::pair<SignatureType, StringAttr>;

/// A ClosureCache stores previously generated closures to prevent duplicate
/// definitions from being generated.
class ClosureCache {
public:
  virtual ~ClosureCache() {}
  virtual StructDeclOp getExisting(ClosureHash key) = 0;
  virtual void storeClosure(ClosureHash key, StructDeclOp closure) = 0;
};

class ClosureEmitter : public StructEmitter {
public:
  ClosureEmitter(LIT::FileModuleOp fileModuleOp, SharedState &shared)
      : StructEmitter(shared), fileModuleOp(fileModuleOp),
        dtorFieldAttr(StringAttr::get(shared.getContext(), "dtor")),
        copyFieldAttr(StringAttr::get(shared.getContext(), "copy")),
        callFieldAttr(StringAttr::get(shared.getContext(), "call")),
        callMethodAttr(
            StringAttr::get(shared.getContext(), "closureCallMethod")) {}

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp createClosureWrapperStructDecl(StringAttr name,
                                              SignatureType signatureType);

  /// Generate a Closure Implementation Struct, a struct that contains the
  /// capture list.
  StructDeclOp replaceNestedFunctionWithClosureImplStructDecl(
      SMLoc loc, ASTDecl &nestedFunctionDecl, ClosureCache &cache);

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
  FileModuleOp fileModuleOp;
  StringAttr dtorFieldAttr;
  StringAttr copyFieldAttr;
  StringAttr callFieldAttr;
  StringAttr callMethodAttr;
};

} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
