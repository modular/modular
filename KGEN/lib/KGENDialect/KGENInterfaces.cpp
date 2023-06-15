//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

LogicalResult impl::verifyCallOp(KGENCallOpInterface op) {
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

  ArrayRef<Type> types = op.getCalleeType().getResultParamTypes();
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
// FuncInterface
//===----------------------------------------------------------------------===//

/// If the specified operation is non-null and contains parameters, collect
/// them into the specified array.
static void collectContextParameters(Operation *op,
                                     SmallVector<ParamDeclAttr> &params) {
  auto decl = dyn_cast_or_null<DeclInterface>(op);
  if (!decl || isa<FuncInterface>(*decl))
    return;
  collectContextParameters(op->getParentOp(), params);
  llvm::append_range(params, decl.getInputParams());
}

/// Return the full signature of this declaration, including parameters from
/// enclosing struct declarations.
SignatureType KGEN::getFullSignature(FuncInterface decl) {
  SignatureType signature = decl.getSignature();

  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  SmallVector<ParamDeclAttr> inputParams;
  collectContextParameters(decl.getOperation()->getParentOp(), inputParams);
  if (inputParams.empty())
    return signature;

  return IndexRefRemapper::prependParams(signature, inputParams);
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.cpp.inc"
