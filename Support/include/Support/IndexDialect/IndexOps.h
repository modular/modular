//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_INDEXDIALECT_INDEXOPS_H
#define SUPPORT_INDEXDIALECT_INDEXOPS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::index {
enum class IndexCmpPredicate : uint32_t;
class IndexCmpPredicateAttr;
} // namespace M::index

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/IndexDialect/Index.h.inc"

#endif // SUPPORT_INDEXDIALECT_INDEXOPS_H
