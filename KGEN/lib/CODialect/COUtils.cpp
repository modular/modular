//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/CODialect/COUtils.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpImplementation.h"

using namespace M;

ParseResult KGEN::parseCoroutineTypes(AsmParser &p, TypeArrayAttr &typeAttr) {
  SmallVector<Type> types;
  if (succeeded(p.parseOptionalColon()) && p.parseCommaSeparatedList([&] {
        return p.parseType(types.emplace_back());
      }))
    return failure();
  typeAttr = TypeArrayAttr::get(p.getContext(), types);
  return success();
}

void KGEN::printCoroutineTypes(AsmPrinter &p, Operation *op,
                               TypeArrayAttr types) {
  if (types.empty())
    return;
  p << " : ";
  llvm::interleaveComma(types, p);
}
