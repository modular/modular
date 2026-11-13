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

#ifndef KGEN_SUPPORT_ATTR_TYPE_MANGLER_H
#define KGEN_SUPPORT_ATTR_TYPE_MANGLER_H

#include "Mojo/KGENDialect/KGENParameters.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/SymbolExport.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"

namespace M::KGEN {

/// This uniquing scheme involves splitting each decl name into a key string
/// and a substring of trailing digits. We track the max of such digits of the
/// same key string and use that to generate the next unique ID.
class MODULAR_CXX_EXPORT NameUniquer {
public:
  NameUniquer(const ParameterUseDefGraph &scope,
              const ParameterUseDefGraph &topLevelGraph);

  /// Check if the name needs mangling.
  bool needsMangling(StringAttr name);

  /// Uniquely mangle a parameter name. Returns the original name if mangling is
  /// not needed.
  StringAttr mangle(StringAttr name);

  /// Update the uniquer with a new name.
  void updateWith(StringRef name);

private:
  /// Update the ids we are tracking with the declarations (including those
  /// nested) in the given scope.
  void updateMaxIds(const ParameterUseDefGraph &scope);

  /// Map to store the maximum id for each base name we are tracking.
  llvm::StringMap<ssize_t> maxIds;

  /// The top level ParameterUseDefGraph that contains nested scopes that that
  /// carry declarations.
  const ParameterUseDefGraph &topLevelGraph;
};

/// Signature types define a nested parameter scope inside a parameter
/// expression. Manually walk and mangle parameter references in attributes and
/// types in an expression tree while accounting for name shadowing in a
/// signature type.
class MODULAR_CXX_EXPORT AttrTypeMangler {
public:
  using Cache = llvm::DenseSet<const void *>;

  /// Create a mangler for the decls in two potentially conflicting scopes.
  AttrTypeMangler(Cache &manglerCache, Builder &b, NameUniquer &uniquer,
                  const llvm::SetVector<StringAttr> &calleeDecls,
                  const ParameterUseDefGraph &topLevelGraph);

  /// Returns true if any declaration needs mangling.
  bool needsMangling() const { return !mangledDecls.empty(); }

  /// Mangle references within a type.
  Type mangleRefsIn(Type type, bool &hasRefs) {
    return mangleRefsInImpl(type, hasRefs);
  }
  Type mangleRefsIn(Type type) {
    bool unused = false;
    return mangleRefsIn(type, unused);
  }

  /// Mangle references within an attribute.
  Attribute mangleRefsIn(Attribute attr, bool &hasRefs);
  Attribute mangleRefsIn(Attribute type) {
    bool unused = false;
    return mangleRefsIn(type, unused);
  }

  /// Optionally mangle a declaration.
  ParamDeclAttr mangleDecl(ParamDeclAttr decl, bool needsMangling);

  /// Mangle attributes and types.
  void mangleElementsIn(Operation *op);

  /// Recursively mangle declarations in the nested scope.
  void recursivelyMangle(Region *scope, const ParameterUseDefGraph &graph);

private:
  template <typename T, typename U = std::conditional_t<
                            std::is_base_of_v<Type, T>, Type, Attribute>>
  U mangleRefsInImpl(T value, bool &hasRefs) {
    const void *key = value.getAsOpaquePointer();
    if (manglerCache.contains(key))
      return value;
    if (const void *mangled = mangledRefs.lookup(key)) {
      hasRefs = true;
      return U::getFromOpaquePointer(mangled);
    }

    SmallVector<Attribute, 16> replAttrs;
    SmallVector<Type, 16> replTypes;
    bool changed = false;
    bool hasNestedRefs = false;
    value.walkImmediateSubElements(
        [&](Attribute attr) {
          Attribute result = mangleRefsIn(attr, hasNestedRefs);
          replAttrs.push_back(result);
          changed |= result != attr;
        },
        [&](Type type) {
          Type result = mangleRefsIn(type, hasNestedRefs);
          replTypes.push_back(result);
          changed |= result != type;
        });

    hasRefs |= hasNestedRefs;
    if (!hasNestedRefs) {
      manglerCache.insert(key);
      return value;
    }
    U result = changed ? value.replaceImmediateSubElements(replAttrs, replTypes)
                       : value;
    mangledRefs.try_emplace(key, result.getAsOpaquePointer());
    return result;
  }

  /// The map of mangled declarations.
  DenseMap<StringAttr, StringAttr> mangledDecls;
  /// A cache of attributes and types known to have no parameter references.
  Cache &manglerCache;
  /// The mangled form of each attribute and type that has a parameter
  /// reference. Uniqued attributes share subexpressions, so this map lets a
  /// walk visit each shared node once. Each mangler owns its map because the
  /// result depends on `mangledDecls`.
  DenseMap<const void *, const void *> mangledRefs;
};

} // namespace M::KGEN

#endif // KGEN_SUPPORT_ATTR_TYPE_MANGLER_H
