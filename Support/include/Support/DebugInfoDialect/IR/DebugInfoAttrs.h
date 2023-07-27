//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/ForwardDecls.h"
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
/// Extract the scope from the location of a function. Functions either have a
/// subprogram scope fused directly to the location, or we consider them as not
/// having any. Therefore this never requires a recursion, and therefore can be
/// done without a location cache.
DISubprogramAttr extractScope(mlir::FunctionOpInterface funcOp);

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

/// If the op has a subprogram scope, update it with the given linkage name
/// (and optionally the given name, if not null), as well as all references to
/// the scope recursively within the body.
void updateSubprogram(mlir::FunctionOpInterface op, StringAttr linkageName,
                      StringAttr name = {});

/// Return the scope from a location of an op within a function's body,
/// recursively walking up through a chain of inlined locations if needed,
/// always following the caller location.
ErrorOr<DebugInfo::DIScopeAttr> getScopeWithinBody(Location loc);

/// Update the location of the op as if it was inlined at the given caller
/// location, handling special location interfaces.
void updateInlinedLoc(Operation *op, Location callerLoc);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
