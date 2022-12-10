//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_SUPPORT_INVOKE_H
#define KGEN_SUPPORT_INVOKE_H

#include "GenericML/Support/Tensor.h"
#include "llvm/Support/Compiler.h"

namespace M::KGEN {

namespace detail {

inline constexpr ssize_t MaximumTensorRank = 5;
using TensorShapeArrayType = std::array<ssize_t, MaximumTensorRank>;
using TensorShapeStorageType = SmallVector<TensorShapeArrayType, 4>;

template <class T>
inline constexpr bool is_arrayref_v = false;

template <class T>
inline constexpr bool is_arrayref_v<ArrayRef<T>> = true;

template <class T>
inline constexpr bool is_arrayref_v<MutableArrayRef<T>> = true;

/// If the type is an ArrayRef, then destructure the ArrayRef into the form that
/// KGEN expects (a tuple of a pointer, size, and dtype).  For all other types,
/// just return the type as is.
template <typename T>
constexpr auto destructure_kgen_arguments(TensorShapeStorageType &shapeStorage,
                                          T &&arg) {
  if constexpr (std::is_same_v<T, Tensor>) {
    ssize_t rank = arg.getShape().getRank();
    assert(rank <= MaximumTensorRank &&
           "KGEN only supports tensors of rank <= 5");
    auto tensorShape = arg.getShape();
    TensorShapeArrayType shape{0};
    std::copy(tensorShape.begin(), tensorShape.begin() + rank, shape.begin());
    shapeStorage.emplace_back(shape);
    // NOTE: Currently cannot distinguish between input and output args,
    // so all buffers are extracted as mutable.
    return std::tuple{arg.getMutableBuffer(), rank,
                      reinterpret_cast<ssize_t *>(shapeStorage.back().data()),
                      arg.getEltType().getValue()};
  } else if constexpr (is_arrayref_v<T>) {
    using array_element_type = std::decay_t<decltype(*arg.data())>;
    return std::tuple{const_cast<array_element_type *>(arg.data()),
                      (ssize_t)arg.size(),
                      DTypeForCXXType<typename T::value_type>::kind.getValue()};
  } else {
    return std::tuple{std::forward<T>(arg)};
  }
}

/// Destructure the arguments (returning a tuple).
template <typename First, typename... Rest>
constexpr auto destructure_kgen_arguments(TensorShapeStorageType &shapeStorage,
                                          First &&first, Rest &&...rest) {
  return std::tuple_cat(
      destructure_kgen_arguments(shapeStorage, std::forward<First>(first)),
      destructure_kgen_arguments(shapeStorage, std::forward<Rest>(rest)...));
}
} // namespace detail

/// Invoke a KGEN kernel with the given arguments. This would perform
/// destructuring so that ArrayRefs are passed as a tuple of (pointer, size,
/// dtype). All other types are passed as is.
template <typename F, typename... Args>
inline auto invoke(F &&f, Args &&...args) {
  detail::TensorShapeStorageType shapeStorage;
  return std::apply(std::forward<F>(f),
                    detail::destructure_kgen_arguments(
                        shapeStorage, std::forward<Args>(args)...));
}

} // namespace M::KGEN

#endif // KGEN_SUPPORT_INVOKE_H
