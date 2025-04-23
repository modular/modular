//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_HLCFDIALECT_HLCFOPS_H
#define KGEN_HLCFDIALECT_HLCFOPS_H

#include "KGEN/HLCFDialect/HLCFInterfaces.h"
#include "KGEN/Interpreter/InterpreterInterface.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "KGEN/HLCFDialect/HLCFAttrs.h"

#define GET_OP_CLASSES
#include "KGEN/HLCFDialect/HLCF.h.inc"

#endif // KGEN_HLCFDIALECT_HLCFOPS_H
