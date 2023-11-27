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
#include "Support/DebugInfoDialect/IR/DebugInfoExprAttrInterfaces.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h.inc"

//===----------------------------------------------------------------------===//
// Support
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {

/// This class provides utilities for prepending conversion expressions to an
/// existing conversion expression. As optimizations accumulate in the IR,
/// the conversion expression of a debuginfo needs to track new optimizations
/// by prepending conversions, i.e. replacing the leaves of expressions with
/// new subtrees.
/// All conversion thru the same prepender instance goes thru a shared attr/type
/// replacer cache.
class DIExprLeafReplacer {
public:
  DIExprLeafReplacer(std::function<ErrorOr<DIExprAttr>(DIType)> conversionFunc);

  // Apply the leafReplacer to the input expression.
  // In practice, this means replacing the leaves of expr with the result of the
  // leafReplacer.
  ErrorOr<DIExprAttr> apply(DIExprAttr expr);

private:
  // Records any error message emitted during an `apply` call.
  // Always cleared before running the replacer.
  std::string currErrorMsg;

  std::function<ErrorOr<DIExprAttr>(DIType)> leafReplacer;
  mlir::AttrTypeReplacer replacer;
};

/// Extract the scope from the location of a function. Functions either have
/// a subprogram scope fused directly to the location, or we consider them
/// as not having any. Therefore this never requires a recursion, and
/// therefore can be done without a location cache.
DISubprogramAttr extractScope(mlir::FunctionOpInterface funcOp);

/// Extract the debug info scope from the location of the given operation.
DIScopeAttr extractScope(Operation *op);

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
                      SourceNameAttr name = {});

/// Return the scope from a location of an op within a function's body,
/// recursively walking up through a chain of inlined locations if needed,
/// always following the caller location.
template <typename ScopeT>
ScopeT extractScopeFrom(Location loc) {
  if (auto fused = loc->findInstanceOf<mlir::FusedLocWith<ScopeT>>())
    return fused.getMetadata();
  return {};
}

/// Update the location of the op as if it was inlined at the given caller
/// location, handling special location interfaces. An optional flag can be
/// specified to indicate that we are in an `always_inline(nodebug)` context,
/// and need to erase the location of the inlined operations by replacing them
/// with the location of the call.
void updateInlinedLoc(Operation *op, Location callerLoc,
                      bool stripDebugInfo = false);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
