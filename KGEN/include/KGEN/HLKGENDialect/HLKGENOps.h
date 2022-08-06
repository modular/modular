//===- HLKGENOps.h --------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the HLKGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_HLKGENOPS_H
#define KGEN_KGENDIALECT_HLKGENOPS_H

#include "KGEN/HLKGENDialect/HLKGENDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"

namespace M::KGEN {
class ReturnOp;
}

#define GET_OP_CLASSES
#include "KGEN/HLKGENDialect/HLKGEN.h.inc"

#endif // KGEN_KGENDIALECT_NLKGENOPS_H
