//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_CODIALECT_COUTILS_H
#define KGEN_CODIALECT_COUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
class TypeArrayAttr;
namespace KGEN {
ParseResult parseCoroutineTypes(AsmParser &p, TypeArrayAttr &typeAttr);
void printCoroutineTypes(AsmPrinter &p, Operation *op, TypeArrayAttr types);
} // namespace KGEN
} // namespace M

#endif // KGEN_CODIALECT_COUTILS_H
