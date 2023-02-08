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

LogicalResult impl::verifyIfTopLevel(DeclInterface decl,
                                     SymbolTableCollection &symtab) {
  if (isa<DeclInterface>(decl->getParentOp()))
    return success();
  for (Region &r : decl->getRegions())
    if (failed(ParameterUseDefGraph(r).verify(symtab)))
      return failure();
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

  ArrayRef<ParamDeclAttr> params = op.getCalleeType().getResultParams();
  if (op.getParamDecls().size() != params.size()) {
    return op->emitOpError("declares ")
           << op.getParamDecls().size() << " result parameters, but callee has "
           << params.size();
  }
  for (auto [decl, result, idx] : llvm::zip(
           op.getParamDecls(), params, llvm::seq<unsigned>(0, params.size()))) {
    if (decl.getType() == result.getType())
      continue;
    return op.emitOpError("result parameter #")
           << idx << " declared with type " << decl.getType()
           << " but callee has " << result.getType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENInterfaces.cpp.inc"
