//===- KGEN/ZAPDialect/ZAPOps.h -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_ZAPDIALECT_ZAPOPS_H
#define KGEN_ZAPDIALECT_ZAPOPS_H

#include "KGEN/MetaDialect/MetaTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class DTypeType;
namespace POP {
class PointerType;
class ScalarType;
} // namespace POP
namespace ZAP {
class BufferType;
} // namespace ZAP
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/ZAPDialect/ZAP.h.inc"

#endif // KGEN_ZAPDIALECT_ZAPOPS_H
