//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ZAPDIALECT_ZAPTYPES_H
#define KGEN_ZAPDIALECT_ZAPTYPES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Types.h"

namespace M::KGEN {
class KGENDType;
namespace POP {
class PointerType;
class SIMDType;
} // namespace POP
} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "KGEN/ZAPDialect/ZAPTypes.h.inc"

#endif // GEN_ZAPDIALECT_ZAPTYPES_H
