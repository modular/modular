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
#include "SharedState.h"
#include "StructEmitter.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {

class ClosureEmitter {
public:
  ClosureEmitter(LIT::FileModuleOp fileModuleOp, Type noneType,
                 SharedState &shared)
      : fileModuleOp(fileModuleOp), noneType(noneType), shared(shared),
        structEmitter(shared),
        dtorFieldAttr(StringAttr::get(shared.getContext(), "dtor")),
        copyFieldAttr(StringAttr::get(shared.getContext(), "copy")),
        moveFieldAttr(StringAttr::get(shared.getContext(), "move")) {}

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp createClosureWrapperStructDecl(StringAttr name,
                                              SignatureType signatureType);
  Type getNoneType() const { return noneType; }
  SharedState &sharedState() const { return shared; }

  /// Generate a Closure Implementation Struct, a struct that contains the
  /// capture list.
  StructDeclOp createClosureImplStructDecl(StringAttr name,
                                           SignatureType closureImplSignature,
                                           unsigned captureCount);

  /// Generate an initializer on the ClosureWrapper that accepts a ClosureImpl
  /// instance.
  LIT::FuncOp createWrapperInitWithImpl(StructDeclOp closureWrapper,
                                        StructDeclOp closureImpl,
                                        SMLoc location);

private:
  FileModuleOp fileModuleOp;
  Type noneType;
  SharedState &shared;
  StructEmitter structEmitter;
  StringAttr dtorFieldAttr;
  StringAttr copyFieldAttr;
  StringAttr moveFieldAttr;
};
} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
