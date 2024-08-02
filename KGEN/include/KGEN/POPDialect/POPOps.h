//===----------------------------------------------------------------------===//
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

#include "KGEN/Interpreter/InterpreterInterface.h"
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPEnums.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class DTypeType;
class PackType;
class PointerType;
class StringType;
class StructType;
class VariadicType;
class VariantType;
} // namespace M::KGEN

namespace M::KGEN::POP {
class CmpPredicateAttr;
class AtomicOrderingAttr;
class AtomicBinOpAttr;
class FastmathFlagsAttr;
class PrefetchTagAttr;
class PrefetchLocalityAttr;

class ArrayType;
class SIMDType;
class UnionType;
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.h.inc"

#endif // KGEN_POPDIALECT_POPOPS_H
