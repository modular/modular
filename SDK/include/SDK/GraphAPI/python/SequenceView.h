//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_SEQUENCEVIEW_H
#define SDK_GRAPHAPI_PYTHON_SEQUENCEVIEW_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "nanobind/nanobind.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

namespace M::Graph::Python {

// A type-erased sequence structure that can be passed to Python
// as an immutable view on (potentially non-contiguous) sequence data in C++.
struct SequenceView {
  void *data;
  size_t len;
  void *(*at)(void *data, size_t index);
  nanobind::handle (*from_cpp)(void *, nanobind::rv_policy,
                               nanobind::detail::cleanup_list *);
  void (*deleter)(void *data) = nullptr;

  template <typename T>
  static nanobind::handle cast(void *ptr, nanobind::rv_policy rv_policy,
                               nanobind::detail::cleanup_list *cleanup) {
    return nanobind::detail::make_caster<T>::from_cpp(*static_cast<T *>(ptr),
                                                      rv_policy, cleanup);
  }

  template <typename T>
  SequenceView(llvm::ArrayRef<T> array)
      : data((void *)array.data()), len(array.size()),
        at([](void *data, size_t index) {
          T *item = &static_cast<T *>(data)[index];
          return static_cast<void *>(item);
        }),
        from_cpp(&cast<T>) {}

  template <typename Range, typename BaseT, typename T, typename PointerT,
            typename ReferenceT>
  SequenceView(llvm::detail::indexed_accessor_range_base<
               Range, BaseT, T, PointerT, ReferenceT> &range)
      : data(std::make_unique<Range>(range).release()), len(range.size()),
        at([](void *data, size_t index) {
          ReferenceT item = static_cast<Range *>(data)->operator[](index);
          if constexpr (std::is_same_v<ReferenceT, PointerT>) {
            return *reinterpret_cast<void **>(&item);
          } else {
            return static_cast<void *>(&item);
          }
        }),
        from_cpp(&cast<T>),
        deleter([](void *data) { delete static_cast<Range *>(data); }) {}

  SequenceView(SequenceView &&other)
      : data{other.data}, len{other.len}, at{other.at},
        from_cpp{other.from_cpp},
        deleter{std::exchange(other.deleter, nullptr)} {}

  SequenceView &operator=(SequenceView &&) = delete;
  SequenceView(const SequenceView &) = delete;
  SequenceView &operator=(const SequenceView &) = delete;

  ~SequenceView() {
    // Don't run deleter if python is being shut down
    if (deleter && Py_IsInitialized())
      deleter(data);
  }

  void *operator[](size_t index) { return at(data, index); }
};

} // namespace M::Graph::Python

#endif // SDK_GRAPHAPI_PYTHON_SEQUENCEVIEW_H
