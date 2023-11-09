//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the ClosureEmitter class.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser/CaptureParameter.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"

#include "KGEN/KGENDialect/KGENParameters.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Casting.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Given an attribute, visit all contained attributes until all parameter
/// declaration references are visited.
static void
recursivelyCallback(Attribute attribute,
                    std::function<void(StringAttr, Type)> callback) {
  if (auto paramDeclRef = dyn_cast<ParamDeclRefAttr>(attribute)) {
    callback(paramDeclRef.getName(), paramDeclRef.getType());
    return;
  }
  attribute.walkImmediateSubElements(
      [&callback](Attribute child) {
        if (auto paramDeclRef = dyn_cast<ParamDeclRefAttr>(child)) {
          callback(paramDeclRef.getName(), paramDeclRef.getType());
        } else {
          recursivelyCallback(child, callback);
        }
      },
      [](Type type) {});
}

/// Given a function and a captured parameter name and type, populate the
/// capturedParams data structure with all implicitly captured parameters. For
/// example, if `alias A = myFunc[B,C]` is declared inside the parent function
/// and the root is "A", then this function should populate capturedParams with
/// {"A", "B", "C"}
static SmallVector<std::pair<StringAttr, ParameterCapture>>
collectImplicitParameterCaptures(LIT::FuncOp parentFunction,
                                 StringAttr rootName, unsigned depth) {
  SmallVector<std::pair<StringAttr, ParameterCapture>> capturedParams;
  ParameterUseDefGraph parameterUseDefGraph(parentFunction.getBodyRegion());
  KGEN::ParameterCollector::Analysis paramCache;
  parameterUseDefGraph.calculate(paramCache);
  SmallVector<StringAttr> worklist{rootName};
  auto addParameterToWorklist = [&](StringAttr paramName, Type paramType) {
    auto existingEntry = std::find_if(
        capturedParams.begin(), capturedParams.end(),
        [paramName](std::pair<StringAttr, ParameterCapture> entry) {
          return entry.first == paramName;
        });
    if (existingEntry == capturedParams.end())
      worklist.push_back(paramName);
  };
  while (!worklist.empty()) {
    StringAttr current = worklist.back();
    worklist.pop_back();
    ParamDefinition definition = parameterUseDefGraph.defs.lookup(current);
    // Collect parameter dependencies from definition.
    if (!definition.defOp)
      continue;
    if (auto funcOp = dyn_cast<LIT::FuncOp>(*definition.defOp)) {
      // Result parameters are not defined by the function.
      ParamDeclAttr paramDecl = funcOp.getInputParams()[definition.index];
      ParameterCapture capture(paramDecl, definition.index, depth);
      capturedParams.push_back({current, capture});
    } else if (auto alias = dyn_cast<AliasDeclOp>(*definition.defOp)) {
      capturedParams.push_back({alias.getParamDecl().getName(),
                                ParameterCapture(alias.getParamDecl(), -1,
                                                 depth, definition.defOp)});
      // Add operands of the alias's expression to the worklist.
      alias.getValue().walkImmediateSubElements(
          [addParameterToWorklist](Attribute immediateChild) {
            recursivelyCallback(immediateChild, addParameterToWorklist);
          },
          [](Type type) {});
    }
  }
  return capturedParams;
}

/// A value that represents where a parameter is defined with respect to a
/// declaration.
enum class ParameterRelation {
  Undefined =
      0,  // The parameter is not defined within the scope of this declaration.
  Input,  // The parameter is declared as an input parameter in the declaration.
  Result, // The parameter is declared as a result parameter in the declaration.
  Local   // The parameter is defined in the body of the declaration.
};

/// Given a parameter and a function, return the relationship of that parameter
/// to that function along with its index in the declaration list, and -1 if the
/// relationship is not input or result. If the srcSpelling is an empty string,
/// the local scope is not searched. Otherwise, we search for local parameter
/// declarations with the name of the source spelling.
static std::pair<ParameterRelation, int>
parameterRelationshipToFunction(SharedState &shared, ASTDecl *functionDecl,
                                ParamDeclRefAttr declRef,
                                StringRef srcSpelling) {
  if (!functionDecl)
    return {ParameterRelation::Undefined, -1};
  auto idxOfParam = [&](ArrayRef<ParamDeclAttr> arrayRef) -> int {
    for (auto [i, entry] : llvm::enumerate(arrayRef))
      if (entry.getName() == declRef.getName())
        return i;
    return -1;
  };

  auto function = dyn_cast<LIT::FuncOp>(*functionDecl);
  if (!function)
    return {ParameterRelation::Undefined, -1};
  int inputIndex = idxOfParam(function.getInputParams());
  if (inputIndex > -1)
    return {ParameterRelation::Input, inputIndex};
  int resultIndex = idxOfParam(function.getResultParams());
  if (resultIndex > -1)
    return {ParameterRelation::Result, resultIndex};

  // Okay, we may have referenced a parameter defined in the body. If this is
  // the case we must search by spelling not the mangled parameter name.
  if (srcSpelling.empty())
    return {ParameterRelation::Undefined, -1};
  LookupResult lookup = shared.lookupAndResolveDecl(
      srcSpelling, functionDecl->getLoc(), *functionDecl, false);
  if (lookup.isSuccess())
    return {ParameterRelation::Local, -1};
  return {ParameterRelation::Undefined, -1};
}

/// CaptureInfo contains all the information required to record a capture.
struct CaptureInfo {
  CaptureInfo(StringRef srcSpelling, unsigned depth,
              ParamDeclRefAttr paramDeclRef, ASTDecl &container)
      : srcSpelling(srcSpelling), depth(depth), paramDeclRef(paramDeclRef),
        container(container) {}
  LogicalResult recordCaptureInFunction(SharedState &shared,
                                        ASTDecl *currentParent,
                                        Location location) const {
    if (!currentParent)
      return LogicalResult::failure();
    auto parentFunction = dyn_cast<LIT::FuncOp>(*currentParent);
    if (!parentFunction)
      return LogicalResult::failure();
    if (!parentFunction.getResultParams().empty()) {
      shared.emitError(
          location, "TODO: Support result parameters and escaping closures.");
      return LogicalResult::failure();
      ;
    }
    auto [paramRelationToParent, index] = parameterRelationshipToFunction(
        shared, currentParent, paramDeclRef, srcSpelling);
    switch (paramRelationToParent) {
    case ParameterRelation::Result:
      [[fallthrough]];
    case ParameterRelation::Input:
      shared.addCapturedParameterToScope(
          container, ParameterCapture(paramDeclRef.getName(),
                                      paramDeclRef.getType(), index, depth));
      break;
    case ParameterRelation::Local: {
      // We have captured a parameter. Collect all the parameters that
      // this parameter depends on.
      SmallVector<std::pair<StringAttr, ParameterCapture>> capturedParams =
          collectImplicitParameterCaptures(parentFunction,
                                           paramDeclRef.getName(), depth);
      for (auto [name, paramCapture] : llvm::reverse(capturedParams))
        shared.addCapturedParameterToScope(container, paramCapture);
      break;
    }
    default:
      break;
    }
    if (paramRelationToParent == ParameterRelation::Undefined)
      return LogicalResult::failure();
    return LogicalResult::success();
  }
  LogicalResult recordCaptureInStruct(SharedState &shared,
                                      ASTDecl *currentParent) const {
    if (!currentParent)
      return LogicalResult::failure();
    auto parentStruct = dyn_cast<StructDeclOp>(*currentParent);
    if (!parentStruct)
      return LogicalResult::failure();
    for (auto [index, inputParam] :
         llvm::enumerate(parentStruct.getInputParams())) {
      if (inputParam.getName() == paramDeclRef.getName()) {
        ParameterCapture capture(inputParam, index, depth);
        shared.addCapturedParameterToScope(container, capture);
        return LogicalResult::success();
      }
    }
    return LogicalResult::failure();
  }

private:
  /// The source spelling of the captured parameter.
  StringRef srcSpelling;
  /// The number of declarations between the parameter declaration and the
  /// capture reference.
  unsigned depth;
  /// The capture attribute.
  ParamDeclRefAttr paramDeclRef;
  /// The ASTDecl of the nested function.
  ASTDecl &container;
};

/// Return the first ASTDecl ancestor that is a function or null if does not
/// exist.
ASTDecl *CaptureUtility::nearestParentFuncOpDecl(ASTDecl &decl,
                                                 bool includeMe) {
  ASTDecl *parentDecl = includeMe ? &decl : decl.getParentDecl();
  while (parentDecl) {
    if (auto function = dyn_cast<LIT::FuncOp>(*parentDecl))
      return parentDecl;
    parentDecl = parentDecl->getParentDecl();
  }
  return nullptr;
}

static LogicalResult recordParameterCaptureWithScopedLookup(
    SharedState &shared, ASTDecl *nestedFunctionDecl, StringRef srcSpelling,
    ParamDeclRefAttr paramDeclRef, Location parameterRefLocation) {
  auto [relationToContainerFuncion, index] = parameterRelationshipToFunction(
      shared, nestedFunctionDecl, paramDeclRef, srcSpelling);
  if (relationToContainerFuncion == ParameterRelation::Undefined) {
    unsigned depth = 0;
    // First parent must be function. Otherwise, it's not a nested function,
    // it is a method.
    ASTDecl *currentParent =
        CaptureUtility::nearestParentFuncOpDecl(*nestedFunctionDecl);
    // Traverse up declarations until the capture is found.
    for (; currentParent; currentParent = currentParent->getParentDecl()) {
      depth++;
      CaptureInfo captureInfo(srcSpelling, depth, paramDeclRef,
                              *nestedFunctionDecl);
      LogicalResult outcomeOfRecordInFunction =
          captureInfo.recordCaptureInFunction(shared, currentParent,
                                              parameterRefLocation);
      if (outcomeOfRecordInFunction.succeeded())
        return outcomeOfRecordInFunction;
      LogicalResult outcomeOfRecordInStruct =
          captureInfo.recordCaptureInStruct(shared, currentParent);
      if (outcomeOfRecordInStruct.succeeded())
        return outcomeOfRecordInStruct;
    }
  }
  return LogicalResult::failure();
}

LogicalResult CaptureUtility::recordParameterCapture(
    SharedState &shared, ASTDecl *nestedFunctionDecl,
    ParamDeclRefAttr paramDeclRef, Location location) {
  return recordParameterCaptureWithScopedLookup(shared, nestedFunctionDecl, "",
                                                paramDeclRef, location);
}

LogicalResult CaptureUtility::recordParameterCapture(
    SharedState &shared, ASTDecl *nestedFunctionDecl, StringRef srcSpelling,
    ParamDeclRefAttr paramDeclRef, Location location) {
  return recordParameterCaptureWithScopedLookup(
      shared, nestedFunctionDecl, srcSpelling, paramDeclRef, location);
}
