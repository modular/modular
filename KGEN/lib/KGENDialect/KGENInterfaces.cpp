//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/ElaboratorOpInterface.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

/// Verify that regions used as signature parameters match in the signature.
static LogicalResult verifyRegionSignatures(KGENCallOpInterface op) {
  auto regionValues =
      llvm::make_filter_range(op.getParamValues(), [](ParamBindAttr value) {
        return value.getValue().isa<ParamCallRegionRefAttr>();
      });

  size_t numRegionParams =
      std::distance(regionValues.begin(), regionValues.end());
  if (numRegionParams != op->getNumRegions())
    return op->emitOpError("expected ")
           << numRegionParams << " body regions but has "
           << op->getNumRegions();

  // Ensure each region parameter matches up in order with the regions.
  for (auto [idx, bind] : llvm::enumerate(regionValues)) {
    auto paramSignature = cast<SignatureType>(bind.getValue().getType());
    Region &region = op->getRegion(idx);
    auto body = cast<FuncInterface>(region.front().getTerminator());
    if (region.front().getOperations().size() != 1)
      return op->emitOpError("expected region #")
             << idx << " to contain only a `kgen.region.body` op";

    if (failed(verifyDeclSignaturesMatch("region", body.getSignature(),
                                         body.getLoc(), "parameter",
                                         paramSignature, op.getLoc())))
      return failure();
  }
  return success();
}

LogicalResult impl::verifyCallOp(KGENCallOpInterface op) {
  if (!op.getCallee())
    return success();

  // Disallow calls from within a concrete function from calling anything with
  // input or output parameters.
  auto func = op->getParentOfType<FuncOp>();
  if (func && !op.getParamValues().empty()) {
    return op.emitOpError("cannot reference generator with input parameters "
                          "from within a concrete 'kgen.func'")
               .attachNote(func.getLoc())
           << "within 'kgen.func' @" << func.getName();
  }

  if (!op.isAllowedInFunc() && func)
    return op.emitOpError("is only allowed in generators pre-elaboration");

  return verifyRegionSignatures(op);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ElaboratorOpInterface.cpp.inc"
#include "KGEN/KGENDialect/KGENInterfaces.cpp.inc"
