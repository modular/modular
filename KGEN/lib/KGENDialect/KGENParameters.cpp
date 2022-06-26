//===- KGENParameters.cpp - KGEN Parameter utilities ----------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for working with KGEN parameters.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Parameter Verification
//===----------------------------------------------------------------------===//

namespace {
struct ParameterVerifier final {
  ParameterVerifier(ParameterDeclsAndUses &parameters)
      : parameters(parameters) {}

  /// Walk the operation and all the operations in its body to find the
  /// definitions and uses of parameters.  This diagnoses and rejects parameter
  /// definitions in invalid positions as well.
  LogicalResult collectParameterDefsAndUses(Operation *topLevelOp);

  /// Once all the defs and uses of parameters are collected, verify that the
  /// uses are correct.
  LogicalResult checkParameterUses();

private:
  /// Scan the specified attribute and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses.
  void collectParameterUsesFromAttr(Attribute attr, Operation *op);

  /// Scan the specified type and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses.
  void collectParameterUsesFromType(Type type, Operation *op);

  /// This is set to true if we find a problem during the collect phase.
  bool hadError = false;

  /// This is the parameter information that we're building.
  ParameterDeclsAndUses &parameters;

  /// Attributes and types are memoized and exist in tree structures with reuse:
  /// naively scanning them can lead to exponential compile time behavior.  As
  /// such, we memoize the attributes and types we've already checked that we
  /// know have no parameters in them.
  llvm::SmallDenseSet<Attribute> parameterLessAttrs;
  llvm::SmallDenseSet<Type> parameterLessTypes;
};
} // end anonymous namespace.

/// Walk the operation and all the operations in its body to find the
/// definitions and uses of parameters.  This diagnoses and rejects parameter
/// definitions in invalid positions as well.
LogicalResult
ParameterVerifier::collectParameterDefsAndUses(Operation *topLevelOp) {
  // TODO: We probably shouldn't walk into IsolatedFromAbove operations.  This
  // walk may need to be adjusted if we have any.
  topLevelOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *bodyOp) {
    Attribute paramDeclsAttr;
    // Scan all the attributes and types to look for uses of parameters.  We let
    // the walker scan the region hierarchy.
    for (const NamedAttribute &namedAttr : bodyOp->getAttrs()) {
      // We handle the `paramDecls` attribute specially, remember it for below.
      if (namedAttr.getName().strref() == "paramDecls") {
        paramDeclsAttr = namedAttr.getValue();
        continue;
      }
      // Scan the attribute tree looking or parameter uses and reject unexpected
      // parameter definitions.
      collectParameterUsesFromAttr(namedAttr.getValue(), bodyOp);
    }

    // Check the types of results to find any parameters embedded in their
    // types.  We don't have to check operands because they are always checked
    // when being defined.
    for (Type type : bodyOp->getResultTypes())
      collectParameterUsesFromType(type, bodyOp);

    // Scan the region list if present.  The walker will automatically recurse
    // for us, but we have to check the block arguments.
    if (bodyOp->getNumRegions()) { // Microoptimization: getRegions() is slow.
      for (auto &region : bodyOp->getRegions()) {
        for (auto &block : region)
          for (Value arg : block.getArguments())
            collectParameterUsesFromType(arg.getType(), bodyOp);
      }
    }

    // Ok, check parameter declarations if present.
    if (!paramDeclsAttr)
      return;

    auto arrayAttr = paramDeclsAttr.dyn_cast<ArrayAttr>();
    if (!arrayAttr) {
      bodyOp->emitError("paramDecls attribute should be an array ")
          << paramDeclsAttr;
      hadError = true;
      return;
    }

    for (Attribute attr : arrayAttr) {
      // All the members of this array must be ParamDeclAttr's.
      auto param = attr.dyn_cast<ParamDeclAttr>();
      if (!param) {
        bodyOp->emitError("unknown attribute kind in paramDecls list ") << attr;
        hadError = true;
        return;
      }

      // We cannot have any redefinitions.
      auto &opAndDeclAttr = parameters.decls[param.getName()];
      if (opAndDeclAttr.first) {
        auto diag = bodyOp->emitError("redeclaration of parameter ")
                    << param.getName();
        diag.attachNote(opAndDeclAttr.first->getLoc())
            << "previous declaration here";
        hadError = true;
        return;
      }
      opAndDeclAttr = {bodyOp, param};
    }
  });

  return failure(hadError);
}

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterVerifier::collectParameterUsesFromAttr(Attribute attr,
                                                     Operation *op) {

  // Collect parameter references.
  if (auto paramRef = attr.dyn_cast<ParamDeclRefAttr>()) {
    parameters.uses.push_back({op, paramRef});
    return;
  }

  // If this attribute has no sub-attributes or we have already scanned it an
  // know that it has no parameters in it, return early.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr>() ||
      parameterLessAttrs.contains(attr))
    return;

  // Reject errant parameter decls.
  if (auto paramDecl = attr.dyn_cast<ParamDeclAttr>()) {
    op->emitError("invalid ParamDeclAttr outside of paramDecls attribute ")
        << paramDecl;
    return;
  }

  size_t oldSize = parameters.uses.size();

  // Otherwise we haven't processed this, check the attribute's type.
  collectParameterUsesFromType(attr.getType(), op);

  // Recursively check for any nested types/attributes, e.g. the elements of an
  // array attribute.
  if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    itf.walkSubElements(
        [&](Attribute attr) { collectParameterUsesFromAttr(attr, op); },
        [&](Type type) { collectParameterUsesFromType(type, op); });
  } else if (attr.isa<DTypeConstantAttr>()) {
    // This attribute doesn't participate with SubElementAttrInterface but we
    // know it doesn't have any subelements.
  } else {
    // Conservatively reject unknown attributes, we don't want someone to forget
    // to conform to SubElementAttrInterface.
    op->emitError("unknown attribute for parameterization scan: ") << attr;
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == parameters.uses.size())
    parameterLessAttrs.insert(attr);
}

/// Scan the specified type and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterVerifier::collectParameterUsesFromType(Type type, Operation *op) {
  // Ignore common trivial types we know are never parameterized, and types we
  // have already scanned.
  if (parameterLessTypes.count(type))
    return;

  size_t oldSize = parameters.uses.size();

  // Recursively check for any nested types, e.g. the input/outputs of a
  // function type.  This also handles types like !meta.scalar etc.
  if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    itf.walkSubElements(
        [&](Attribute attr) { collectParameterUsesFromAttr(attr, op); },
        [&](Type type) { collectParameterUsesFromType(type, op); });
  } else {
    // These are known leaf types that don't participate with
    // SubElementTypeInterface.
    if (!type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>()) {
      // Conservatively reject unknown types, we don't want someone to forget to
      // conform to SubElementTypeInterface.
      op->emitError("unknown type for parameterization scan: ") << type;
    }
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == parameters.uses.size())
    parameterLessTypes.insert(type);
}

/// Once all the defs and uses of parameters are collected, verify that the
/// uses are correct.
LogicalResult ParameterVerifier::checkParameterUses() {
  for (auto [usingOp, paramRefAttr] : parameters.uses) {
    // Check the use is referring to a parameter that was defined.
    auto decl = parameters.decls[paramRefAttr.getName()];
    if (!decl.first) {
      usingOp->emitError("invalid use of parameter with no declaration ")
          << paramRefAttr.getName();
      return failure();
    }

    // Check that the types of the uses match the defs.
    if (decl.second.getType() != paramRefAttr.getType()) {
      auto diag = usingOp->emitError("reference to parameter ")
                  << paramRefAttr.getName() << " with incorrect type "
                  << paramRefAttr.getType();
      diag.attachNote(decl.first->getLoc())
          << "parameter defined with type " << decl.second.getType();
      return failure();
    }

    // FIXME: Check partial ordering.
  }
  return success();
}

/// Collect information about the parameter definitions and uses in the
/// specified operation.  This emits an error and returns `None` on an IR
/// verification error.
Optional<ParameterDeclsAndUses>
ParameterDeclsAndUses::calculate(Operation *topLevelOp) {
  ParameterDeclsAndUses result;
  ParameterVerifier verifier(result);

  // Start by doing a pass over the operation and all the operations in its body
  // to find the definitions and uses of parameters.
  if (failed(verifier.collectParameterDefsAndUses(topLevelOp)))
    return None;

  // Ok, now that we know the set of parameters we have to process, verify that
  // the uses match up and that we have a proper partial order relationship
  // between of definitions and uses.
  if (failed(verifier.checkParameterUses()))
    return None;

  return std::move(result);
}
