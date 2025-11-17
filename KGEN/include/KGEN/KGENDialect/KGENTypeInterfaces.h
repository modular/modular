//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPEINTERFACES_H
#define KGEN_KGENDIALECT_KGENTYPEINTERFACES_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

namespace M::KGEN {
class SymTabEvaluationContext;
enum class SugarKind : uint32_t;
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENTypeInterfaces.h.inc"

#endif // KGEN_KGENDIALECT_KGENTYPEINTERFACES_H
