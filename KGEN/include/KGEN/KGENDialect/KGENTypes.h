//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares types for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_H
#define KGEN_KGENDIALECT_KGENTYPES_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENDialect.h"
#include "KGEN/KGENDialect/KGENTypeInterfaces.h"
#include "Support/ForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"

namespace M::KGEN {
class FuncInterface;
class ParamBindAttr;
class ParamBindArrayAttr;
class ParamDeclArrayAttr;
class TypeArrayAttr;
class VariadicType;
class VariadicAttr;

/// Utility class for remapping named parameter references to index references.
class IndexRefRemapper {
public:
  /// Populate the remapper with named input and result parameters.
  IndexRefRemapper(ArrayRef<ParamDeclAttr> inputParams,
                   ArrayRef<ParamDeclAttr> resultParams, size_t offset = 0);

  /// Remap a value.
  template <typename T>
  T remap(T value) {
    constexpr bool isType = std::is_base_of_v<Type, T>;
    std::conditional_t<isType, Type, Attribute> result;
    if constexpr (isType)
      result = remapTypeImpl(value);
    else
      result = remapAttrImpl(value);
    return cast<T>(result);
  }

private:
  /// Remap an attribute.
  Attribute remapAttrImpl(Attribute attr);
  /// Remap a type.
  Type remapTypeImpl(Type type);

  /// Walk and remap values.
  template <typename T>
  auto normalizeSignatureWalk(T value, size_t depth = 0)
      -> std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute>;

  /// Mapping from parameter reference to an index and `isResult` flag.
  DenseMap<StringAttr, std::pair<size_t, bool>> mapping;
  /// The index offset of references to root input parameters.
  size_t offset;
};

} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "KGEN/KGENDialect/KGENTypes.h.inc"

#endif // KGEN_KGENDIALECT_KGENTYPES_H
