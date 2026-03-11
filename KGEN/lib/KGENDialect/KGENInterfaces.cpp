//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/Diagnostics/DiagnosticEmitter.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/Compiler/OperationUtils.h"

using namespace M;
using namespace KGEN;
using namespace KGEN::Diag;

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
      mlir::InFlightDiagnostic diag = emitError(
          op, Diag::DiagID::err_cannot_reference_generator_input_parameters);
      attachNote(diag, func.getLoc(), Diag::DiagID::note_within_kgen_func,
                 func.getName());
      return diag;
    }

    if (!op.isAllowedInFunc())
      return KGEN::Diag::emitOpError(
          op, Diag::DiagID::err_only_allowed_generators_pre_elaboration);
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
      return KGEN::Diag::emitError(
          op, Diag::DiagID::err_c_exported_lacks_export_symbol);
    if (!isCIdentifier(exportName)) {
      return emitError(op,
                       Diag::DiagID::err_export_c_function_invalid_identifier,
                       exportName.getValue());
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.cpp.inc"
