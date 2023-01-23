//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_ASYNCVALUEREF_H
#define LLCL_RUNTIME_ASYNCVALUEREF_H

#include "LLCL/Runtime/AsyncValue.h"

namespace M::LLCL {

/// This class is a typed smart pointer that automatically maintains the
/// reference count and static type for an underlying AsyncValue object.
///
/// It is analogous to AnyAsyncValueRef, but provides AsyncValue specific
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

  AsyncValueRef(AnyAsyncValueRef &&value) : value(std::move(value)) {}
  AsyncValueRef(AsyncValueRef &&rhs) : value(std::move(rhs.value)) {}

  AsyncValueRef &operator=(AsyncValueRef &&rhs) {
    value = std::move(rhs.value);
    return *this;
  }

  // Support implicit conversion from AsyncValueRef<Derived> to
  // AsyncValueRef<Base>.
  template <typename DerivedT,
            std::enable_if_t<std::is_base_of<T, DerivedT>::value, int> = 0>
  AsyncValueRef(AsyncValueRef<DerivedT> &&u) : value(u.ReleaseRCRef()) {}

  // Allow implicit conversion to type-erased AnyAsyncValueRef
  operator AnyAsyncValueRef() && { return std::move(value); }

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
  static AsyncValueRef<T> allocate(CompactRuntimePtr runtime) {
    return AsyncValue::allocate<T>(runtime);
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

  /// Return a raw pointer to the AsyncValue.
  AsyncValue *getPointer() const { return value.getPointer(); }

  T &operator*() const {
    assert(value && "null AsyncValueRef");
    return value->get<T>();
  }

  T *operator->() const {
    assert(value && "null AsyncValueRef");
    return &value->get<T>();
  }

  /// Take ownership of the underlying pointer away from the AsyncValueRef and
  /// reset it to null.
  AsyncValue *release() { return value.release(); }

  /// Take ownership of the underlying pointer away from the AsyncValueRef and
  /// reset it to null.
  AnyAsyncValueRef releaseRCRef() { return std::move(value); }

  // Make an explicit copy of this AsyncValueRef, increasing value's refcount
  // by one.
  AsyncValueRef<T> copy() const { return AsyncValueRef(copyRCRef()); }

  // Make a copy of value, increasing value's refcount by one.
  AnyAsyncValueRef copyRCRef() const { return value.copy(); }

  /// Manually drop the reference in this AsyncValueRef, setting it to null.
  void reset() { value.reset(); }

  /// Test for null.
  explicit operator bool() const { return getPointer() != nullptr; }

  //===--------------------------------------------------------------------===//
  // Core AsyncValue operations
  //===--------------------------------------------------------------------===//

  CompactRuntimePtr getRuntime() const { return value->getRuntime(); }

  /// Return true if this has been turned into an error.
  bool isError() const { return value->isError(); }

  /// Return the stored value in an `available` AsyncValue.
  T &get() const { return value->get<T>(); }

  /// Construct the payload of a ConcreteAsyncValue and change its state to
  /// `available`. Requires that the AsyncValue's state is `unconstructed`.
  /// This reference is consumed just before any downstream waiters are
  /// triggered. See AsyncValue::emplace for more details.
  template <typename... Args>
  void emplace(Args &&...args) && {
    AsyncValue::emplace<T, Args...>(std::move(value),
                                    std::forward<Args>(args)...);
  }

  /// Construct the payload of a ConcreteAsyncValue and change its state to
  /// `available`.  Requires that the AsyncValue's state is `unconstructed`.
  template <typename... Args>
  void emplace(Args &&...args) const & {
    value->emplace<T>(std::forward<Args>(args)...);
  }

  /// Mark an "unconstructed" AsyncValue as an error.
  void setToError(EncodedDiagnostic diagnostic) const {
    value->setToError(std::move(diagnostic));
  }

  using Waiter = AsyncValue::Waiter;
  using ConsumingWaiter = llvm::unique_function<void(AsyncValueRef<T> &&ref)>;

  /// Perform an 'andThen' operation on this AsyncValueRef. This reference
  /// is consumed in order to be made available to the waiter.
  /// See AsyncValue::andThen for more details.
  template <bool IsAsync>
  void andThen(ConsumingWaiter &&waiter) && {
    // TODO(#7399): The compiler and runtime are happy without this eta
    // expansion of waiter -- check it's doing the right thing and remove.
    AsyncValue::andThen<IsAsync>(
        std::move(value),
        [waiter = std::move(waiter)](AnyAsyncValueRef &&ref) mutable {
          waiter(AsyncValueRef<T>(std::move(ref)));
        });
  }

  void andThenSync(ConsumingWaiter &&waiter) && {
    std::move(*this).template andThen</*IsAsync=*/false>(std::move(waiter));
  }

  void andThenAsync(ConsumingWaiter &&waiter) && {
    std::move(*this).template andThen</*IsAsync=*/true>(std::move(waiter));
  }

  /// Perform an 'andThen' operation on this AsyncValueRef.
  template <bool IsAsync>
  void andThen(Waiter &&waiter) {
    getPointer()->template andThen<IsAsync>(std::move(waiter));
  }

  void andThenSync(Waiter &&waiter) {
    andThen</*IsAsync=*/false>(std::move(waiter));
  }

  void andThenAsync(Waiter &&waiter) {
    andThen</*IsAsync=*/true>(std::move(waiter));
  }

private:
  AnyAsyncValueRef value;
};

//===----------------------------------------------------------------------===//
// AsyncValueRefWithEncodedLocation
//===----------------------------------------------------------------------===//

/// This template may be used where it is useful to bundle together a reference
/// to an AsyncValue (either AnyAsyncValueRef or AsyncValueRef<T>) with an
/// EncodedLocation.
///
/// This value is larger than an AsyncValue reference (3 words instead of 1) and
/// involves more reference counting (EncodedLocations need to keep their
/// decoder alive), so it should only be used where needed.
template <typename AVRefType>
class AsyncValueRefWithEncodedLocation : public AVRefType {
public:
  AsyncValueRefWithEncodedLocation(AVRefType refValue, EncodedLocation loc)
      : AVRefType(std::move(refValue)), loc(std::move(loc)) {}

  AsyncValueRefWithEncodedLocation(AsyncValueRefWithEncodedLocation &&) =
      default;

  /// Fill this AsyncValue with an error that has the specified message.
  void setToError(Error message) const {
    this->getPointer()->setToError({std::move(message), loc.copy()});
  }

  /// Provide access to the location.
  const EncodedLocation &getLocation() const { return loc; }

private:
  EncodedLocation loc;
};
} // namespace M::LLCL

#endif // LLCL_RUNTIME_ASYNCVALUEREF_H
