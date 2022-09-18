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

enum class GeneratorOrFuncKind {
  func,
  generator,
  interface,

  // HLKGEN dialect
  hlgenerator,
};

/// Parse the MLIR syntax for a kgen.generator, kgen.func and related
/// operators.
ParseResult parseGeneratorOrFunc(OpAsmParser &parser, OperationState &result,
                                 GeneratorOrFuncKind opKind);
void printGeneratorOrFunc(OpAsmPrinter &p, mlir::FunctionOpInterface op);

/// Verify that a list of parameter declarations from a generator or func
/// matches those of an interface.  This produces an error diagnostic and
/// returns failure when a problem is detected, or returns true if everything is
/// ok.
ParseResult verifyParameterList(ParamDeclArrayAttr originatorParamDecls,
                                ParamDeclArrayAttr targetParamDecls,
                                const char *originatorName,
                                Location originatorLoc, const char *targetName,
                                Location targetLoc, const char *parameterKind);

/// Check that the specified generator/interfaces matches signature
/// information with the other interface.
LogicalResult verifyDeclMatchesInterface(const char *originatorName,
                                         KGENDeclInterface originatorDecl,
                                         const char *interfaceName,
                                         GeneratorInterfaceOp interfaceDecl);

/// Check that the specified declaration signatures match, checking the
/// parameter and value type information.
LogicalResult verifyDeclSignaturesMatch(
    const char *originatorName, ParamDeclArrayAttr originatorInputParams,
    ParamDeclArrayAttr originatorResultParams, FunctionType originatorType,
    Location originatorLoc, const char *interfaceName,
    ParamDeclArrayAttr targetInputParams, ParamDeclArrayAttr targetResultParams,
    FunctionType targetType, Location targetLoc);

} // namespace M::KGEN

#define GET_OP_CLASSES
#include "KGEN/KGENDialect/KGEN.h.inc"

#endif // KGEN_KGENDIALECT_KGENOPS_H
