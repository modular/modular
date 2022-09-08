//===- KGEN/KGENDialect/KGENOps.h -----------------------------------------===//
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

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDeclInterface.h"
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

enum class GeneratorOrKernelKind {
  kernel,
  generator,
  interface,

  // HLKGEN dialect
  hlgenerator,
};

/// Parse the MLIR syntax for a kgen.generator, kgen.kernel and related
/// operators.
ParseResult parseGeneratorOrKernel(OpAsmParser &parser, OperationState &result,
                                   GeneratorOrKernelKind opKind);
void printGeneratorOrKernel(OpAsmPrinter &p, mlir::FunctionOpInterface op);

/// Verify that a list of parameter declarations from a generator or kernel
/// matches those of an interface.  This produces an error diagnostic and
/// returns failure when a problem is detected, or returns true if everything is
/// ok.
ParseResult verifyParameterList(ArrayRef<ParamDeclAttr> originatorParamDecls,
                                ArrayRef<ParamDeclAttr> interfaceParamDecls,
                                const char *originatorName,
                                mlir::FunctionOpInterface originatorDecl,
                                const char *interfaceName,
                                GeneratorInterfaceOp interfaceDecl,
                                const char *parameterKind);

/// Check that the specified generator/interfaces matches signature
/// information with the other interface.
LogicalResult verifyDeclMatchesInterface(
    const char *originatorName, mlir::FunctionOpInterface originatorDecl,
    const char *interfaceName, GeneratorInterfaceOp interfaceDecl);

} // namespace M::KGEN

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.h.inc"

#endif // KGEN_KGENDIALECT_KGENOPS_H
