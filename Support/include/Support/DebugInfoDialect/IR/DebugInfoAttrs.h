//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

namespace mlir {
class FunctionOpInterface;
}

namespace M::DebugInfo {
/// This class represents the base class of all DebugInfo attributes.
class DIAttr : public Attribute {
public:
  using Attribute::Attribute;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents the base class of DebugInfo attributes that form
/// a scope.
class DIScopeAttr : public DIAttr {
public:
  using DIAttr::DIAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};

/// This class represents the base class of DebugInfo attributes that form
/// a local scope.
class DILocalScopeAttr : public DIScopeAttr {
public:
  using DIScopeAttr::DIScopeAttr;

  /// Support LLVM type casting.
  static bool classof(Attribute attr);
};
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/IR/DebugInfoEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
/// Extract a debug info scope from the given location.
DIScopeAttr extractScope(Location loc);
/// Extract the debug info scope from the location of the given operation.
DIScopeAttr extractScope(Operation *op);
template <typename ScopeAttrT, typename T>
ScopeAttrT extractScope(T value) {
  return dyn_cast_or_null<ScopeAttrT>(extractScope(value));
}

/// This class represents an attribute/type replacer with proper defaults for
/// updating debug information within operations.
class DIAttrTypeReplacer : public mlir::AttrTypeReplacer {
public:
  /// TODO: Upstream this templated version to AttrTypeReplacer.
  template <typename T, typename U>
  T replace(U value) {
    return dyn_cast_if_present<T>(replace(value));
  }
  using mlir::AttrTypeReplacer::replace;

  /// Replace elements within the given operation.
  void replaceElementsIn(Operation *op);

  /// Replace elements within the given operation, and any nested operations.
  void recursivelyReplaceElementsIn(Operation *op);
};

/// If the op has a subprogram scope, change the name and linkage name to that
/// given, and replace all nested subprogram attributes recursively with it.
template <typename OpTy>
void renameSubprogramsInScopes(StringAttr name, OpTy op) {
  auto sp = DebugInfo::extractScope<DebugInfo::DISubprogramAttr>(op);
  if (!sp)
    return;

  DebugInfo::DISubprogramAttr newSp = sp.cloneWith(name, name);
  DebugInfo::DIAttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DebugInfo::DISubprogramAttr attr) { return newSp; });
  replacer.recursivelyReplaceElementsIn(op);
}

/// Verify that a function-like op has the correct location scope. Succeeds if
/// the location has no scope attached to it.
LogicalResult verifyFuncLocScope(mlir::FunctionOpInterface funcOp);

} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
