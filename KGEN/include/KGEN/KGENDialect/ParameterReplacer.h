//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PARAMETERREPLACER_H
#define KGEN_PARAMETERREPLACER_H

#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ParameterReplacer
//===----------------------------------------------------------------------===//

/// This class is an attribute and type sub-element replacer that is aware of
/// the current parameter scope. It is useful for working with index parameter
/// references.
template <typename DerivedT>
class ParameterReplacer {
public:
  /// Remap a value.
  template <typename T>
  T replace(T value) {
    return cast<T>(replaceImpl(value, /*depth=*/0));
  }

  /// Remap a range of values.
  template <typename T>
  SmallVector<T> replace(ArrayRef<T> values) {
    return llvm::map_to_vector(values, [&](T value) { return replace(value); });
  }

protected:
  template <typename T>
  std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute>
  replaceImpl(T value, size_t depth) {
    if (!value)
      return nullptr;

    // These are common leaf attributes that we know are never parameterized.
    if constexpr (std::is_base_of_v<Type, T>) {
      if (isa<TypeType, DTypeType, StringType, IntLiteralType, KGEN::NoneType,
              TargetType, BuildInfoType, IntegerType, FloatType>(value))
        return value;
    } else {
      if (isa<NoneAttr, IntegerAttr, FloatAttr, DTypeConstantAttr,
              IntLiteralAttr, TargetParamAttr, MLIROpAttr>(value))
        return value;
    }

    // If we've already processed this value, just reuse the memoized result.
    auto it = rewritten.find({depth, value.getAsOpaquePointer()});
    if (it != rewritten.end())
      return decltype(replaceImpl(value, depth))::getFromOpaquePointer(
          it->second);

    // Don't cache null results.
    auto result = getDerived()->doReplace(value, depth);
    if (!result)
      return nullptr;

    rewritten.try_emplace({depth, value.getAsOpaquePointer()},
                          result.getAsOpaquePointer());
    return result;
  }

private:
  DerivedT *getDerived() { return static_cast<DerivedT *>(this); }

  /// Depth-aware cache from original attribute or type to rewritten attribute
  /// or type and remembers complex values that haven't been rewritten (noted as
  /// being mapped to themselves).
  DenseMap<std::pair<size_t, const void *>, const void *> rewritten;
};

//===----------------------------------------------------------------------===//
// IndexParameterReplacer
//===----------------------------------------------------------------------===//

/// A subclass of parameter replacer that contains even more common logic for
/// working with index references.
template <typename DerivedT>
class IndexParameterReplacer
    : public ParameterReplacer<IndexParameterReplacer<DerivedT>> {
  template <typename T>
  std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute>
  doReplace(T value, size_t depth) {
    if (auto result = static_cast<DerivedT *>(this)->tryReplace(value, depth))
      return result;

    if constexpr (std::is_base_of_v<Type, T>) {
      if (isa<ParameterScopeTypeInterface>(value))
        ++depth;
    } else {
      if (isa<ParameterScopeAttrInterface>(value))
        ++depth;
    }

    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    bool changed = false;
    auto walkFn = [&](auto value, SmallVectorImpl<decltype(value)> &values) {
      auto newValue = this->replaceImpl(value, depth);
      changed |= newValue != value;
      values.push_back(newValue);
    };
    value.walkImmediateSubElements(
        [&](Attribute attr) { walkFn(attr, newAttrs); },
        [&](Type type) { walkFn(type, newTypes); });
    if (!changed)
      return value;
    return value.replaceImmediateSubElements(newAttrs, newTypes);
  }

  friend class ParameterReplacer<IndexParameterReplacer<DerivedT>>;
};

} // namespace M::KGEN

#endif // KGEN_PARAMETERREPLACER_H
