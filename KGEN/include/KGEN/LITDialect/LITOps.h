//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_LITOPS_H
#define KGEN_KGENDIALECT_LITOPS_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class NoneType;
class ReturnOp;

namespace POP {
class PointerType;
}

enum class SpecialFunctionKind {
  // This is not a special function.  This enumerator should always have value
  // zero so it can be used as a false condition in an if.
  kNormal = 0,

  kInit = 1, //< __init__
  kNew = 2,  //< __new__
};

} // namespace M::KGEN

#define GET_OP_CLASSES
#include "KGEN/LITDialect/LIT.h.inc"

#endif // KGEN_KGENDIALECT_NLKGENOPS_H
