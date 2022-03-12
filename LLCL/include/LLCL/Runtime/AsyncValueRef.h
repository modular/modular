//===- LLCL/Runtime/AsyncValueRef.h ---------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_ASYNCVALUEREF_H
#define LLCL_RUNTIME_ASYNCVALUEREF_H

#include "LLCL/Runtime/AsyncValue.h"

namespace LLCL {

/// This class is a typed smart pointer that automatically maintains the
/// reference count and static type for an underlying AsyncValue object.
///
/// It is analogous to RCRef<AsyncValue>, but provides AsyncValue specific
/// helper methods, and doesn't require passing <T> to get() or emplace().
/// It follows the design of RCRef, including not being implicitly copyable.
///
template <typename T>
class AsyncValueRef {
public:
  //===--------------------------------------------------------------------===//
  // Smart Pointer operations
  //===--------------------------------------------------------------------===//

  AsyncValueRef() = default;
  ~AsyncValueRef() = default;

  AsyncValueRef(RCRef<AsyncValue> &&value) : value(std::move(value)) {}
  AsyncValueRef(AsyncValueRef &&rhs) : value(std::move(rhs.value)) {}

  // Support implicit conversion from AsyncValueRef<Derived> to
  // AsyncValueRef<Base>.
  template <typename DerivedT,
            std::enable_if_t<std::is_base_of<T, DerivedT>::value, int> = 0>
  AsyncValueRef(AsyncValueRef<DerivedT> &&u) : value(u.ReleaseRCRef()) {}

  // Allow implicit conversion to type-erased RCRef<AsyncValue>
  operator RCRef<AsyncValue>() && { return std::move(value); }

  /// This constructor forms a reference to the specified pointer, increasing
  /// the underlying reference count by 1.
  static AsyncValueRef<T> copy(AsyncValue *pointer) {
    AsyncValueRef<T> ref;
    ref.value = LLCL::copyRCRef(pointer);
    return ref;
  }

  /// This constructor forms a reference to the specified pointer, taking
  /// ownership it, and thus not increasing the reference count.
  static AsyncValueRef<T> take(AsyncValue *pointer) {
    AsyncValueRef<T> ref;
    ref.value = LLCL::takeRCRef(pointer);
    return ref;
  }

  /// Create an AsyncValue for the specified type in "unconstructed" state.
  /// This should be `emplace`'d, `construct`'d, or finalized with an error.
  static AsyncValueRef<T> createUnconstructed(CompactRuntimePtr runtime) {
    return AsyncValue::createUnconstructed<T>(runtime);
  }

  /// Create an AsyncValue for the specified type in "constructed" but non-ready
  /// state.  When This should be `markReady()`, or finalized with an error.
  template <typename... Args>
  static AsyncValueRef<T> createConstructed(CompactRuntimePtr runtime,
                                            Args &&...args) {
    return AsyncValue::createConstructed<T>(runtime,
                                            std::forward<Args>(args)...);
  }

  /// Create an AsyncValue for the specified type in "available" and ready
  /// state. This is a terminal state for an AsyncValue, it can never change out
  /// of this state.
  template <typename... Args>
  static AsyncValueRef<T> createReady(CompactRuntimePtr runtime,
                                      Args &&...args) {
    return AsyncValue::createReady<T>(runtime, std::forward<Args>(args)...);
  }

  //===--------------------------------------------------------------------===//
  // Smart Pointer operations
  //===--------------------------------------------------------------------===//

  // Return a raw pointer to the AsyncValue.
  AsyncValue *getPointer() const { return value.getPointer(); }

  AsyncValue &operator*() const {
    assert(value && "null AsyncValueRef");
    return *getPointer();
  }

  AsyncValue *operator->() const {
    assert(value && "null AsyncValueRef");
    return getPointer();
  }

  /// Take ownership of the underlying pointer away from the AsyncValueRef and
  /// reset it to null.
  AsyncValue *release() { return value.release(); }

  /// Take ownership of the underlying pointer away from the AsyncValueRef and
  /// reset it to null.
  RCRef<AsyncValue> releaseRCRef() { return std::move(value); }

  // Make an explicit copy of this AsyncValueRef, increasing value's refcount
  // by one.
  AsyncValueRef<T> copy() const { return AsyncValueRef(copyRCRef()); }

  // Make a copy of value, increasing value's refcount by one.
  RCRef<AsyncValue> copyRCRef() const { return value.copy(); }

  /// Manually drop the reference in this AsyncValueRef, setting it to null.
  void reset() { value.reset(); }

  /// Test for null.
  explicit operator bool() const { return getPointer() != nullptr; }

  //===--------------------------------------------------------------------===//
  // Core AsyncValue operations
  //===--------------------------------------------------------------------===//

  /// Return the stored value in an `available` AsyncValue.
  T &get() const { return value->get<T>(); }

  /// Construct the payload of a ConcreteAsyncValue and change its state to
  /// `available`.  Requires that the AsyncValue's state is `unconstructed`.
  template <typename... Args>
  void emplace(Args &&...args) const {
    value->emplace<T>(std::forward<Args>(args)...);
  }

private:
  RCRef<AsyncValue> value;
};
} // namespace LLCL

#endif // LLCL_RUNTIME_ASYNCVALUEREF_H
