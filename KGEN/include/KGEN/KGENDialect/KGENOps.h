//===- KGENOps.h ----------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation classes for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENOPS_H
#define KGEN_KGENDIALECT_KGENOPS_H

#include "KGEN/KGENDialect/KGENDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace M::KGEN {
class ParamDeclAttr;
class GeneratorInterfaceOp;

LogicalResult verifyDeclMatchesInterface(
    const char *originatorName, mlir::FunctionOpInterface originatorDecl,
    const char *interfaceName, GeneratorInterfaceOp interfaceDecl);

enum class GeneratorOrKernelKind {
  kernel,
  generator,
  interface,

  // HLKGEN dialect
  hlgenerator,
};

/// Given an arbitrary MLIR operation, classify it into a declaration kind or
/// return None if unknown.
Optional<GeneratorOrKernelKind> classifyDecl(Operation *op);

/// Parse the MLIR syntax for a kgen.generator, kgen.kernel and related
/// operators.
ParseResult parseGeneratorOrKernel(OpAsmParser &parser, OperationState &result,
                                   GeneratorOrKernelKind opKind);
void printGeneratorOrKernel(OpAsmPrinter &p, mlir::FunctionOpInterface op);

} // namespace M::KGEN

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.h.inc"

#endif // KGEN_KGENDIALECT_KGENOPS_H
