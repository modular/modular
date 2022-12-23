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

#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::KGEN {
class ListType;
class TypeArrayAttr;
} // namespace M::KGEN

namespace M::KGEN::POP {
enum class CmpPredicate : uint32_t;
class CmpPredicateAttr;
enum class PrefetchTag : uint32_t;
class PrefetchTagAttr;
enum class PrefetchLocality : uint32_t;
class PrefetchLocalityAttr;

class ArrayType;
class ClosureType;
class PointerType;
class ScalarType;
class SIMDType;
class StructType;
class VariantType;
} // namespace M::KGEN::POP

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "KGEN/POPDialect/POP.h.inc"

#endif // KGEN_POPDIALECT_POPOPS_H
