//===- KGEN/POPDialect/POPOps.h -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the Meta dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPOPS_H
#define KGEN_POPDIALECT_POPOPS_H

#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN::POP {
enum class CmpPredicate : uint32_t;
class CmpPredicateAttr;
class PointerType;
class ScalarType;
class SIMDType;
class StructType;
} // namespace M::KGEN::POP

#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.h.inc"

#endif // KGEN_POPDIALECT_POPOPS_H
