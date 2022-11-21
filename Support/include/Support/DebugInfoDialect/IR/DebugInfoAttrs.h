//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
#define SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"

//===----------------------------------------------------------------------===//
// DIAttr
//===----------------------------------------------------------------------===//

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

#endif // SUPPORT_DEBUGINFODIALECT_IR_DEBUGINFOATTRS_H
