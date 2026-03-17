//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LOWERLIT_LOWERLITTYPES_H
#define KGEN_LOWERLIT_LOWERLITTYPES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"

namespace mlir {
class LockedSymbolTableCollection;
} // namespace mlir

namespace M::KGEN::LIT {

struct StructDecl {
  /// Return true if the struct should be flattened when lowered.
  /// Single-element register-passable structs are flattened to their element
  /// type, UNLESS @align is specified with a value > 1 - in that case we
  /// preserve the struct to maintain the alignment metadata.
  bool isSingleElement() const {
    if (!isRegisterPassable || fields.size() != 1)
      return false;
    if (!minAlignment)
      return true;
    // An alignment of 1 means no explicit alignment.
    if (auto intAttr = dyn_cast<IntegerAttr>(minAlignment))
      return intAttr.getInt() == 1;
    return false;
  }

  bool isSinglePointerElement() const {
    return isSingleElement() && isa<PointerType>(fields.front().second);
  }

  /// The un-parameterized SourceNameAttr for the struct decl.
  DebugInfo::SourceNameAttr sourceName;
  /// The struct input parameters.
  ParamDeclArrayAttr decls;
  /// True if the type is register-passable.
  bool isRegisterPassable;
  /// Explicit minimum alignment specified via @align(N), or null if
  /// unspecified. Uses TypedAttr to support future parametric alignment.
  TypedAttr minAlignment;
  /// The location of the decl, for emitting errors.
  LocationAttr loc;
  /// The field names and types of the struct in order.
  SmallVector<std::pair<StringAttr, Type>> fields;
  /// The symbol ref for the type-value generator.
  SymbolRefAttr symRef;

  /// Flags for tracking recursion during DFS.
  bool visited = false, done = false;
};

struct StructDecls {
  LogicalResult process(ModuleOp module, SymbolTable &symtab);

  /// Lookup a struct decl.
  StructDecl &get(StringAttr name) {
    auto it = structDecls.find(name);
    assert(it != structDecls.end() && "struct decl not found");
    return it->second;
  }

  /// A map from struct name and field name to index. Used for lowering `insert`
  /// and `extract` ops.
  DenseMap<std::pair<StringAttr, StringAttr>, int> fieldIndices;
  /// Map from struct name to the lowering info.
  llvm::MapVector<StringAttr, StructDecl> structDecls;
};

LogicalResult lowerLITTypes(ModuleOp module, StructDecls &decls,
                            mlir::LockedSymbolTableCollection &symtab);

} // namespace M::KGEN::LIT

#endif // KGEN_LOWERLIT_LOWERLITTYPES_H
