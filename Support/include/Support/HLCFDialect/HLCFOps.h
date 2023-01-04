//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_HLCFDIALECT_HLCFOPS_H
#define SUPPORT_HLCFDIALECT_HLCFOPS_H

#include "Support/HLCFDialect/HLCFInterfaces.h"
#include "Support/Interpreter/InterpreterInterface.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Support/HLCFDialect/HLCF.h.inc"

#endif // SUPPORT_HLCFDIALECT_HLCFOPS_H
