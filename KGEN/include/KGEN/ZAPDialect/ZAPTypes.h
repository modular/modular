//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ZAPDIALECT_ZAPTYPES_H
#define KGEN_ZAPDIALECT_ZAPTYPES_H

#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/SubElementInterfaces.h"
#include "mlir/IR/Types.h"

namespace M {
class DType;
namespace KGEN::POP {
class PointerType;
class SIMDType;
} // namespace KGEN::POP
} // namespace M

#define GET_TYPEDEF_CLASSES
#include "KGEN/ZAPDialect/ZAPTypes.h.inc"

#endif // GEN_ZAPDIALECT_ZAPTYPES_H
