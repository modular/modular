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

namespace M::KGEN::LIT {
class ClosureEmitter {
public:
  ClosureEmitter(LIT::FileModuleOp fileModuleOp) : fileModuleOp(fileModuleOp) {}

  /// Generate a Closure Wrapper Struct, a struct that contains an opaque
  /// pointer to the underlying Closure Implementation instance.
  StructDeclOp createClosureWrapperStructDecl(StringAttr name,
                                              Location location,
                                              SignatureType signatureType);

private:
  FileModuleOp fileModuleOp;
};
} // namespace M::KGEN::LIT

#endif // CLOSUREEMITTER_H
