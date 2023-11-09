//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PARAMETERREPLACER_H
#define KGEN_PARAMETERREPLACER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Types.h"
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
    if (getDerived()->isKnownLeaf(value))
      return value;

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

} // namespace M::KGEN

#endif // KGEN_PARAMETERREPLACER_H
