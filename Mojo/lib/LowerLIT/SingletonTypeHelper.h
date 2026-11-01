//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef KGEN_LOWERLIT_SINGLETONTYPEHELPER_H
#define KGEN_LOWERLIT_SINGLETONTYPEHELPER_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace M::KGEN::LIT {

struct StructDecl {
  /// Return true if the struct should be flattened when lowered.
  /// Single-element register-passable structs are flattened to their element
  /// type, UNLESS @align is specified with a value > 1 - in that case we
  /// preserve the struct to maintain the alignment metadata.
  /// True when the struct is unconditionally register-passable (isMemoryOnly
  /// is a concrete false).
  bool isRegisterPassable() const {
    if (auto b = dyn_cast<BoolAttr>(isMemoryOnlyAttr))
      return !b.getValue();
    return false;
  }

  bool isSingleElement() const {
    if (!isRegisterPassable() || fields.size() != 1)
      return false;
    if (!minAlignment)
      return true;
    if (auto intAttr = dyn_cast<IntegerAttr>(minAlignment))
      return intAttr.getInt() == 1;
    return false;
  }

  /// The un-parameterized SourceNameAttr for the struct decl.
  DebugInfo::SourceNameAttr sourceName;
  /// The struct input parameters.
  ParamDeclArrayAttr decls;
  /// The isMemoryOnly attribute for the KGEN struct type. Can be a concrete
  /// BoolAttr or an i1-typed constraint proposition for conditional RP.
  TypedAttr isMemoryOnlyAttr;
  /// Explicit minimum alignment specified via @align(N), or null if
  /// unspecified. Uses TypedAttr to support future parametric alignment.
  TypedAttr minAlignment;
  /// The location of the decl, for emitting errors.
  LocationAttr loc;
  /// The field names and types of the struct in order.
  SmallVector<std::pair<StringAttr, Type>> fields;
  /// The symbol ref for the type-value generator.
  SymbolRefAttr symRef;
};

struct StructDecls {
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

/// Flatten the given symbol reference, collapsing all nested scopes into one
/// mangled name.
FlatSymbolRefAttr flattenSymbolRefAttr(SymbolRefAttr ref);
/// Helper class for identifying singleton types.
/// Currently only considers a struct type as singleton if for all possible
/// parameter bindings, the struct type always yields a singleton type.
/// Supporting parametric singleton struct types is overkill right now.
class SingletonTypeHelper {
public:
  SingletonTypeHelper(mlir::ModuleOp &module, mlir::SymbolTable &symtab,
                      StructDecls &processedStructs)
      : module(module), symtab(symtab), processedStructs(processedStructs) {}

  bool isSingletonType(mlir::Type type);

  /// If `type` is a singleton type, this returns the singleton value of that
  /// type. Otherwise, returns a null result.
  mlir::TypedAttr getSingletonValue(mlir::Type type);

private:
  using StructCacheTy = llvm::DenseMap<
      mlir::StringAttr,
      llvm::SmallVector<std::tuple<mlir::StringAttr, mlir::TypedAttr>>>;

  /// Returns the iterator for the referenced struct decl into
  /// `alwaysSingletonStructs` (or end() if the decl is not always a singleton).
  StructCacheTy::iterator lookupStructSingletonFields(mlir::SymbolRefAttr ref);

  /// Determine the singleton-ness of a struct decl. This function always
  /// inserts the key into exactly one of:
  /// - alwaysSingletonStructs
  /// - notAlwaysSingletonStructs
  /// Returns the iterator for the key into `alwaysSingletonStructs` (or end()
  /// if it wasn't inserted there).
  /// This function is run at most once for each struct decl in the IR.
  StructCacheTy::iterator populateStructSingletonFields(
      mlir::StringAttr key,
      llvm::ArrayRef<std::pair<mlir::StringAttr, mlir::Type>> fields);

  mlir::ModuleOp module;
  /// The SymbolTable contains all not-yet processed struct decls (unflattened),
  /// while StructDecls contains all struct decls already processed.
  mlir::SymbolTable &symtab;
  StructDecls &processedStructs;
  /// The keys are the struct decls that are always singletons (regardless
  /// of parameter bindings, if applicable). Each key is mapped to a list of
  /// singleton values that make up the singleton value of this struct. The
  /// exact singleton value in the form of a TypedAttr may differ depending
  /// on parameter bindings to the struct decl, so it cannot be produced
  /// until a concrete LIT::StructType is passed in.
  StructCacheTy alwaysSingletonStructs;
  /// These are the struct decls that are known to not always be singletons.
  llvm::DenseSet<mlir::StringAttr> notAlwaysSingletonStructs;

  /// Ephemeral data. Tracks the struct decls currently being checked in
  /// recursive `getSingletonValue` calls. If the same struct decl ever occurs
  /// again, it is not checked again, but immediately returns as _not_ a
  /// singleton type (since it is actually illegal, and will be caught later).
  llvm::DenseSet<mlir::StringAttr> inProgressStructs;
};

} // namespace M::KGEN::LIT

#endif // KGEN_LOWERLIT_SINGLETONTYPEHELPER_H
