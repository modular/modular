//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines an KGEN MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_H
#define KGEN_KGENDIALECT_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/StringMap.h"

// Pull in the dialect definition.
#include "KGEN/KGENDialect/KGENDialect.h.inc"

namespace M::KGEN {
template <typename TypeT>
void KGENDialect::registerMnemonicType() {
  registerPrettyType(
      TypeT::getMnemonic(), +[](AsmParser &p) { return TypeT::parse(p); },
      mlir::TypeID::get<TypeT>(),
      +[](AsmPrinter &p, Type type) {
        p << TypeT::getMnemonic();
        cast<TypeT>(type).print(p);
      });
}
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_H
