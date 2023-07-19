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
#include "Support/DebugInfoDialect/IR/DIBuilder.h"

namespace M::KGEN::LIT {
class ClosureEmitter {
public:
  ClosureEmitter(LIT::FileModuleOp fileModuleOp, Type noneType,
                 SharedState *shared)
      : fileModuleOp(fileModuleOp), noneType(noneType), shared(shared) {}

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp createClosureWrapperStructDecl(StringAttr name,
                                              Location location,
                                              SignatureType signatureType);
  Type getNoneType() const { return noneType; }
  SharedState *sharedState() const { return shared; }

private:
  FileModuleOp fileModuleOp;
  Type noneType;
  SharedState *shared;
};
} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
