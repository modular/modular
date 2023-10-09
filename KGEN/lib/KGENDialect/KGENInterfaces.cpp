//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/Compiler/OperationUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyGeneratorUser(GeneratorUserOpInterface op) {
  if (!op.getCallee())
    return success();

  // Disallow calls from within a concrete function from calling anything with
  // input or output parameters.
  if (auto func = op->getParentOfType<FuncOp>()) {
    auto symbolCst = dyn_cast<SymbolConstantAttr>(op.getCallee());
    if (!symbolCst || !symbolCst.getParamValues().empty()) {
      return op.emitOpError("cannot reference generator with input parameters "
                            "from within a concrete 'kgen.func'")
                 .attachNote(func.getLoc())
             << "within 'kgen.func' @" << func.getName();
    }

    if (!op.isAllowedInFunc())
      return op.emitOpError("is only allowed in generators pre-elaboration");
  }

  // Verify the result parameter types if the signature is known.
  auto sig = op.getCalleeSignature();
  if (!sig)
    return success();

  ArrayRef<Type> types = sig.getResultParamTypes();
  if (op.getParamDecls().size() != types.size()) {
    return op->emitOpError("declares ")
           << op.getParamDecls().size() << " result parameters, but callee has "
           << types.size();
  }
  for (auto [decl, type, idx] : llvm::zip(
           op.getParamDecls(), types, llvm::seq<unsigned>(0, types.size()))) {
    if (decl.getType() == type)
      continue;
    return op.emitOpError("result parameter #")
           << idx << " declared with type " << decl.getType()
           << " but callee has " << type;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExportInterface
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyExportInterface(Operation *op) {
  auto itf = cast<ExportInterface>(op);
  if (itf.isCExported()) {
    StringAttr exportName = itf.getLinkageNameAttr();
    if (!exportName)
      return op->emitOpError("is C exported but lacks an export symbol alias");
    if (!isCIdentifier(exportName)) {
      return mlir::emitError(
                 op->getLoc(),
                 "C exported function name is not a valid C identifier, "
                 "allowed characters: [a-zA-Z0-9_]: ")
             << exportName.getValue();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.cpp.inc"
