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

/// Return the scope from a location of an op within a function's body,
/// recursively walking up through a chain of inlined locations if needed,
/// always following the caller location.
static ErrorOr<DebugInfo::DIScopeAttr> getScopeWithinBody(Location loc) {
  DebugInfo::DIScopeAttr scope;
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    // FusedLoc _may_ contain the scope. If it doesn't, we need to ensure that
    // all the fused locations have the same scope, which we extract.
    scope = dyn_cast_or_null<DebugInfo::DIScopeAttr>(fusedLoc.getMetadata());
    if (ArrayRef<Location> nestedLocs = fusedLoc.getLocations();
        !scope && !nestedLocs.empty()) {
      UNWRAP_ERROR_OR_SET(scope, getScopeWithinBody(nestedLocs.back()));
      for (Location nestedLoc : nestedLocs.drop_back()) {
        UNWRAP_ERROR(nestedScope, getScopeWithinBody(nestedLoc));
        if (nestedScope != scope)
          return Error("contains inconsistent scopes in fused location");
      }
    }
  }

  // If not dealing with an inlined location, we return a scope (if found).
  auto callSiteLoc = dyn_cast<mlir::CallSiteLoc>(loc);
  if (!callSiteLoc)
    return scope;

  // Otherwise, we walk up the inlining chain.
  return getScopeWithinBody(callSiteLoc.getCaller());
}

LogicalResult impl::verifyFunctionLike(FunctionLike op) {
  if (failed(HLCF::verifyControlFlow(op)))
    return failure();

  // If the function doesn't contain a location scope, we don't verify anything.
  auto fusedLoc =
      dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(op->getLoc());
  if (!fusedLoc)
    return success();

  DebugInfo::DIScopeAttr scope = fusedLoc.getMetadata();
  auto funcScope = dyn_cast<DebugInfo::DISubprogramAttr>(scope);
  if (!funcScope) {
    return op.emitOpError("must have subprogram scope in location, but got ")
           << scope;
  }

  // We walk pre-order, and skip nested functions.
  WalkResult res =
      op.getBodyRegion().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<FunctionLike>(op))
          return WalkResult::skip();

        ErrorOr<DebugInfo::DIScopeAttr> scopeOr =
            getScopeWithinBody(op->getLoc());
        if (scopeOr.isError()) {
          res = op->emitOpError(scopeOr.getError());
          return WalkResult::interrupt();
        }

        // We might find a lexical block scope, so we look through it.
        while (auto lexBlock =
                   dyn_cast_or_null<DebugInfo::DILexicalBlockAttr>(*scopeOr))
          scopeOr = lexBlock.getScope();

        if (funcScope != *scopeOr) {
          res = op->emitOpError("location scope does not match scope of parent "
                                "func location: ")
                << funcScope;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  return failure(res.wasInterrupted());
}

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
