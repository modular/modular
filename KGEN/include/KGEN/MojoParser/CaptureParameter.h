//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Capture Parameter.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_CAPTUREPARAMETER_H
#define KGEN_MOJOPARSER_CAPTUREPARAMETER_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/MojoParser/SharedState.h"

namespace M::KGEN::LIT {

/// ParameterCapture represents a nested function value whose declaration is in
/// the parent function.
class ParameterCapture {
public:
  ParameterCapture(ParamDeclAttr declAttr, int index, unsigned depth,
                   Operation *definingOp = nullptr)
      : paramDeclAttr(declAttr), index(index), depth(depth),
        definingOp(definingOp) {}
  ParameterCapture(StringAttr name, Type type, int index, unsigned depth,
                   Operation *definingOp = nullptr)
      : paramDeclAttr(ParamDeclAttr::get(name, type)), index(index),
        depth(depth), definingOp(definingOp) {}
  StringAttr getName() const { return paramDeclAttr.getName(); }
  Type getType() const { return paramDeclAttr.getType(); }
  int getIndex() const { return index; }
  unsigned getDepth() const { return depth; }
  bool operator<(ParameterCapture const &rhs) const {
    if (depth == rhs.depth)
      return index < rhs.index;
    return depth > rhs.depth;
  }
  Operation *getDefiningOp() const { return definingOp; }
  bool isInputOrResultParameter() const { return definingOp == nullptr; };

private:
  /// The declaration of the captured parameter.
  ParamDeclAttr paramDeclAttr;
  /// The index of the parameter in its declaration list. -1 if the parameter is
  /// a locally declared parameter.
  int index;
  /// The number of struct or func declarations from the capture to the
  /// parameter declaration.
  unsigned depth;
  /// DefiningOp is the operation that declares and defines the parameter. It is
  /// null if the parameter is defined in an Input or Result parameter list of a
  /// struct or function.
  Operation *definingOp;
};

/// The OrderedCaptures exposes the captures in the order in which they
/// were declared in the parent without allowing write access to users.
struct OrderedCaptures {
  OrderedCaptures(SmallVector<ParameterCapture> &capturedParams)
      : orderedCaptures(capturedParams) {}
  auto find(StringAttr key) const {
    return std::find_if(
        orderedCaptures.begin(), orderedCaptures.end(),
        [&](ParameterCapture const &other) { return other.getName() == key; });
  }
  SmallVector<ParameterCapture>::const_iterator begin() const {
    return orderedCaptures.begin();
  }
  SmallVector<ParameterCapture>::const_iterator end() const {
    return orderedCaptures.end();
  }

private:
  SmallVector<ParameterCapture> &orderedCaptures;
};

/// Interface for requesting that a parameter reference be recorded as a capture
/// if the reference and the declaration live in different declaration scopes.
class CaptureUtility {
public:
  static ASTDecl *nearestParentFuncOpDecl(ASTDecl &decl,
                                          bool includeMe = false);

  /// Record a parameter capture, include searching local scope.
  static LogicalResult recordParameterCapture(SharedState &shared,
                                              ASTDecl *nestedFunctionDecl,
                                              StringRef srcSpelling,
                                              ParamDeclRefAttr paramDeclRef,
                                              Location parameterRefLocation);

  /// Record a parameter capture, do not search local scope.
  static LogicalResult recordParameterCapture(SharedState &shared,
                                              ASTDecl *nestedFunctionDecl,
                                              ParamDeclRefAttr paramDeclRef,
                                              Location parameterRefLocation);
};
} // namespace M::KGEN::LIT
#endif // KGEN_MOJOPARSER_CAPTUREPARAMETER_H
