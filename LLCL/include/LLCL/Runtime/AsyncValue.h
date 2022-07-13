//===- LLCL/Runtime/AsyncValue.h ------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares AsyncValue, a lightweight and generic "future" type that
// can be fulfilled by an asynchronously provided value or an error.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_ASYNCVALUE_H
#define LLCL_RUNTIME_ASYNCVALUE_H

#include "LLCL/Runtime/CompactRuntimePtr.h"
#include "LLCL/Support/Diagnostic.h"
#include "Support/AlignedAlloc.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/PointerIntPair.h"

namespace LLCL {
class Runtime;
class WaiterListNode;
class AsyncValue;
namespace Detail {
template <typename T>
class ConcreteAsyncValue;
}

/// AnyAsyncValueRef is a using declaration that keeps typed and untyped
/// reference counted references to AsyncValue more syntactically similar.
using AnyAsyncValueRef = RCRef<AsyncValue>;

/// This is a future of the specified value type. Arbitrary C++ types may be
/// used here, even non-copyable types and expensive ones like "your database".
/// All AsyncValues are allocated out of a specific `Runtime` instance and can
/// identify them with `getRuntime()`.
///
/// An AsyncValue is in one of four states (unconstructed, constructed,
/// available, error), where the first two are considered "non-ready" and the
/// last two are considered "ready" (waiters are notified).  If it is in the
/// non-ready state, it may have a list of waiters which are notified when the
/// value transitions to a ready state.
//
/// AsyncValue has two possible representations, depending on whether the
/// creator knows the ultimate payload type or not.  If so, we use the
/// ConcreteAsyncValue<T> subclass, which stores the metadata and payload data
/// consecutively (reducing allocations and improving cache effectiveness).  If
/// not, the more general IndirectAsyncValue class adds a level of indirection
/// that allows the payload type to be resolved later.
///
class AsyncValue {
public:
  /// Type registration - AsyncValue requires that each static type be
  /// registered ahead of their use in an AsyncValue.  This method is efficient
  /// in the case where a type is already registered, so it is fine to register
  /// types without guarding against duplicates etc.
  template <typename T>
  static void registerType();

  /// Helper function that calls registerType() for each type in the list.
  template <typename... Ts>
  static void registerTypes();

  //===--------------------------------------------------------------------===//
  // Static creation methods for AsyncValue's
  //===--------------------------------------------------------------------===//

  /// Create an AsyncValue for the specified type in an "unconstructed" state.
  /// This should be `emplace`'d, `construct`'d, or finalized with an error.
  template <typename T>
  static AnyAsyncValueRef allocate(CompactRuntimePtr runtime);

  /// Create an AsyncValue for the specified type in "constructed" but non-ready
  /// state.  When This should be `markReady()`, or finalized with an error.
  template <typename T, typename... Args>
  static AnyAsyncValueRef createConstructed(CompactRuntimePtr runtime,
                                            Args &&...args);

  /// Create an AsyncValue for the specified type in "available" and ready
  /// state. This is a terminal state for an AsyncValue, it can never change out
  /// of this state.
  template <typename T, typename... Args>
  static AnyAsyncValueRef createReady(CompactRuntimePtr runtime,
                                      Args &&...args);

  /// Create an IndirectAsyncValue that may be filled in with any AsyncValue in
  /// the future.
  static AnyAsyncValueRef createIndirect(CompactRuntimePtr runtime);

  /// Create an AsyncValue that has already been turned into an error with the
  /// specified message.
  static AnyAsyncValueRef createError(CompactRuntimePtr runtime,
                                      EncodedDiagnostic diagnostic);

  //===--------------------------------------------------------------------===//
  // State change methods.
  //===--------------------------------------------------------------------===//

  /// Construct the payload of a ConcreteAsyncValue and change its state to
  /// `kConstructed`.  Requires that the AsyncValue's state is `kUnconstructed`,
  /// and is moved to a ready state with `markReady()`.
  template <typename T, typename... Args>
  void construct(Args &&...args);

  /// Construct the payload of a ConcreteAsyncValue and change our state to
  /// `kAvailable`.  Requires that this AsyncValue's state is `kUnconstructed`.
  template <typename T, typename... Args>
  void emplace(Args &&...args);

  /// Mark an "unconstructed" AsyncValue as an error.
  void setToError(EncodedDiagnostic diagnostic);

  /// Transition a "constructed" AsyncValue to "available" and notify any
  /// waiters.
  void markReady() {
    auto oldState = notifyReady(State::kAvailable, nullptr);
    assert(oldState == State::kConstructed &&
           "can only mark 'constructed' values ready");
    (void)oldState;
  }

  /// Resolve an IndirectAsyncValue to point to the specified new value,
  /// resolving any waiters whenever newValue becomes ready.
  void resolveIndirect(AnyAsyncValueRef newValue);

  /// Resolve an IndirectAsyncValue to contain a concrete AsyncValue with a
  /// newly initialized value, resolving any waiters.
  template <typename T, typename... Args>
  void emplaceIndirect(Args &&...args);

  //===--------------------------------------------------------------------===//
  // Primary interface to AsyncValue for clients to use.
  //===--------------------------------------------------------------------===//

  /// Return the `Runtime` instance this is part of.
  CompactRuntimePtr getRuntime() const { return runtime; }

  /// Call the specified closure if the value is ready.  Otherwise, add it
  /// to the waiter list and calls it when the value becomes ready.
  template <typename WaiterT>
  auto andThen(WaiterT &&waiter) -> decltype(waiter(), void());

  /// Call the specified closure if the value is ready.  Otherwise, add it
  /// to the waiter list and calls it when the value becomes ready.  This
  /// overload passes the current value back into the closure as a
  /// `const AnyAsyncValueRef &`.  This eliminates the need to capture the
  /// receiver in the closure and reduces reference count traffic.
  template <typename WaiterT>
  auto andThen(WaiterT &&waiter)
      -> decltype(waiter(AnyAsyncValueRef()), void());

  /// Return the stored value as type T.
  ///
  ///  This requires that the AsyncValue is either constructed or is a fully
  ///  concrete value, and that T be the exact type (or a base type) of the
  ///  actual payload type. When T is a base type of the payload type, the
  ///  following additional conditions are required:
  ///
  ///     1) Both the payload type and T are polymorphic (have virtual function)
  ///        or neither are.
  ///     2) The payload type does not use multiple inheritance.
  ///
  /// The above conditions are required since we store the value at a fixed
  /// from the start of AsyncValue. Violation of either 1) or 2) requires
  /// additional pointer adjustments to get the proper pointer for the base
  /// type, which we do not have sufficient information to perform at runtime.
  template <typename T>
  const T &get() const;

  // Same as the const overload of get(), for mutable use-cases.
  template <typename T>
  T &get() {
    return const_cast<T &>(static_cast<const AsyncValue *>(this)->get<T>());
  }

  /// Return true if this AsyncValue is "Ready" and filled with a concrete
  /// value.   get() will return a value in this state.
  bool isValueAvailable() const { return getState() == State::kAvailable; }

  /// Return true if the AsyncValue is "Ready" and either filled with a concrete
  /// value or an error.
  bool isReady() const { return isReady(getState()); }

  /// Return true if the AsyncValue is fulfilled with an error state.
  bool isError() const { return getState() == State::kError; }

  /// Return the Diagnostic in this AsyncValue, aborting if it isn't an error.
  const EncodedDiagnostic &getDiagnostic() const {
    auto *result = getDiagnosticIfPresent();
    assert(result && "AsyncValue doesn't hold an error");
    return *result;
  }

  /// Return the Diagnostic in this AsyncValue, aborting if it isn't an error.
  EncodedDiagnostic takeDiagnostic() {
    auto *result = getDiagnosticIfPresent();
    assert(result && "AsyncValue doesn't hold an error");
    return std::move(*result);
  }

  /// If this AsyncValue holds an error, return its diagnostic.  If not, return
  /// nullptr.
  const EncodedDiagnostic *getDiagnosticIfPresent() const {
    return const_cast<AsyncValue *>(this)->getDiagnosticIfPresent();
  }

  /// If this AsyncValue holds an error, return its diagnostic.  If not, return
  /// nullptr.
  EncodedDiagnostic *getDiagnosticIfPresent();

  //===--------------------------------------------------------------------===//
  // Type Related functionality
  //===--------------------------------------------------------------------===//

  /// Return a type identifier for the payload held by this AsyncValue.  This is
  /// not set for IndirectAsyncValue's until they are resolved to a value.
  uint16_t getTypeID() const { return typeID; }

  // Return the ID of the given type. Note that at most 2^16-2 (approx. 64K)
  // unique types can be used in AsyncValues, since the ID is 16 bits, and 0 and
  // 2^16-1 are not allowed to be used as type IDs.
  template <typename T>
  static uint16_t getTypeID() {
    return Detail::ConcreteAsyncValue<T>::staticTypeID;
  }

  template <typename T>
  bool isType() const {
    return getTypeID<T>() == typeID;
  }

  /// If this AsyncValue is constructed with the specified C++ type, return a
  /// pointer to the value, otherwise return null.
  template <typename T>
  const T *dyn_cast() const {
    return isType<T>() ? &get<T>() : nullptr;
  }

  /// If this AsyncValue is constructed with the specified C++ type, return a
  /// pointer to the value, otherwise return null.
  template <typename T>
  T *dyn_cast() {
    return isType<T>() ? &get<T>() : nullptr;
  }

  //===--------------------------------------------------------------------===//
  // Low Level Interfaces
  //===--------------------------------------------------------------------===//

  /// This enum indicates whether the AsyncValue was created as a
  /// ConcreteAsyncValue or IndirectAsyncValue.  It is never mutable.
  enum class SubclassKind : uint8_t {
    kConcrete = 0, // ConcreteAsyncValue
    kIndirect = 1, // IndirectAsyncValue
  };

  SubclassKind getSubclassKind() const { return subclassKind; }

  // The state of AsyncValue.  This is mutable as the value evolves.
  enum class State : uint8_t {
    /// The payload's constructor has not been invoked so the value is not
    /// ready for consumption. This state can transition to `kConstructed`,
    /// `kUnconstructedInlineWaiterConstructing`, `kAvailable` and `kError`.
    kUnconstructed = 0,

    /// These two states are the same as kUnconstructed in terms of state (the
    /// payload is not initialized), but demarcate that payload field is being
    /// used to hold the first waiter in the waiter list.  The first enum value
    /// is used when initialization of the waiter starting (the payload field is
    /// claimed by "andThen") the second value is used when the waiter is fully
    /// initialized.
    ///
    /// This state can transition to `kConstructed`, `kAvailable` and `kError`.
    kUnconstructedInlineWaiterConstructing = 1,
    kUnconstructedInlineWaiterPresent = 2,

    /// The payload's constructor is called but the value is not ready for
    /// consumption (triggering waiters). This state can transition to
    /// `kAvailable` and `kError`.
    kConstructed = 3,

    /// The underlying value is constructed and ready for consumption by
    /// waiters and contains an initialized value. This state can not transition
    /// to any other state.
    kAvailable = 4,

    /// This AsyncValue is ready and contains an error, along with an
    /// uninitialized value. This state can not transition to any other state.
    kError = 5,
  };

  /// Return the current state of this AsyncValue.
  State getState() const {
    return waitersAndState.load(std::memory_order_acquire).getInt();
  }

  /// Return true if the specified AsyncValue state is ready, which means the
  /// waiters have all been notiveid.
  static bool isReady(State state) {
    return state == State::kAvailable || state == State::kError;
  }
  static bool isConstructedOrAvailable(State state) {
    return state == State::kConstructed || state == State::kAvailable;
  }

  /// Return true if reference count is 1.
  bool isUnique() const { return refcount.load() == 1; }

  /// Return true if we tracking of live AsyncValue instances is enabled.
  static bool isAllocationTrackingEnabled() {
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
    return true;
#else
    // Only track the number of alive AsyncValue instances in debug builds.
    return false;
#endif
  }

  /// Return the total number of async values that are currently live in the
  /// process. This is intended for debugging/assertions only, and shouldn't be
  /// used for mainline logic in the runtime.
  static ssize_t getNumAllocatedInstances() {
    assert(isAllocationTrackingEnabled() &&
           "AsyncValue instance tracking disabled!");
    return totalAllocatedAsyncValues.load(std::memory_order_relaxed);
  }

  /// AsyncValue maintains a list of waiters that are waiting for notification
  /// that this value transitioned to Available or Error.
  using Waiter = llvm::unique_function<void(const AnyAsyncValueRef &arg)>;

private:
  // Reference counting, only accessible to RCRef<>.
  template <typename T>
  friend class RCRef;

  /// Increase the reference count.
  void addRef();
  void addRef(uint16_t count);

  /// Decrease the reference count of this object, potentially deallocating it.
  void dropRef(uint16_t count = 1);

  //===--------------------------------------------------------------------===//
  // State held by an AsyncValue
  //===--------------------------------------------------------------------===//

  /// This is the number of individual users of the AsyncValue, when it drops
  /// to zero, the AsyncValue is deallocated.
  std::atomic<int32_t> refcount{1};

  /// This is a compact (8-bit) pointer to the enclosing Runtime instance.
  const CompactRuntimePtr runtime;

  /// Whether this is an indirect or concrete AsyncValue.
  const SubclassKind subclassKind : 1;

  /// hasVTable has the same value for a given payload type T.
  const bool hasVTable : 1;

  // NOTE: 6 unused padding bits.

  /// This is a 16-bit value that identifies the type.  This is dynamically set
  /// for IndirectAsyncValue's when they get resolved.
  uint16_t typeID;

protected:
  struct WaiterListNodePointerTraits {
    static inline void *getAsVoidPointer(WaiterListNode *ptr) { return ptr; }
    static inline WaiterListNode *getFromVoidPointer(void *ptr) {
      return static_cast<WaiterListNode *>(ptr);
    }
    enum { NumLowBitsAvailable = 3 };
  };

  /// The waiter list and the state are compacted into a single atomic word,
  /// since the fields need to be accessed at the same time for state changes.
  ///
  /// Invariant: If the state is ready, then the waiter list must be nullptr.
  using WaitersAndState = llvm::PointerIntPair<WaiterListNode *, 3, State,
                                               WaiterListNodePointerTraits>;

  std::atomic<WaitersAndState> waitersAndState;

protected:
  M::LogicalResult moveState(WaitersAndState &oldValue, State newState);
  void runWaitersAndDeallocate(WaiterListNode *list);
  WaitersAndState andThenOutOfLine(Waiter waiter, WaitersAndState oldValue);
  void destroyWithRefCountZero();
  State notifyReady(State newState, llvm::Optional<Waiter> *extraWaiter);

  /// Invoke a single waiter immediately.
  template <typename WaiterCallable>
  void runOneWaiter(WaiterCallable &waiter) {
    // We pass the AsyncValue in as a `const AnyAsyncValueRef&` to make the
    // ownership very clear (they can use the value but have to copy it if
    // persisting it).  We do this delicately to avoid additional refcount
    // bumps.
    auto rcThisRef = AnyAsyncValueRef::take(this);
    waiter(const_cast<const AnyAsyncValueRef &>(rcThisRef));
    (void)rcThisRef.release();
  }

protected:
  /// This layout of this class is designed very carefully to ensure alignment
  /// of the payload to 16 bytes and we don't want to change this.  That said,
  /// we do put the 16 bytes to work (including metadata about the concrete
  /// type of the value, whether vtables exist or not, etc) in order to detect
  /// common programmer mistakes quickly.
  static constexpr int kAsyncValueSize = 16;

  AsyncValue(SubclassKind subclassKind, State state, bool hasVTable,
             uint16_t typeID, CompactRuntimePtr runtime)
      : runtime(runtime), subclassKind(subclassKind), hasVTable(hasVTable),
        typeID(typeID), waitersAndState(WaitersAndState(nullptr, state)) {
    if (isAllocationTrackingEnabled())
      ++totalAllocatedAsyncValues;
  }

  ~AsyncValue() {
    if (isAllocationTrackingEnabled())
      --totalAllocatedAsyncValues;
  }

private:
  AsyncValue(const AsyncValue &) = delete;
  void operator=(const AsyncValue &) = delete;

  /// This is a global counter of the number of AsyncValue instances currently
  /// live in the process.  This is intended to be used for debugging only, and
  /// is only kept in sync if `isAllocationTrackingEnabled()` returns true.
  static std::atomic<ssize_t> totalAllocatedAsyncValues;
};

//===----------------------------------------------------------------------===//
// ConcreteAsyncValue implementation.
//===----------------------------------------------------------------------===//

namespace Detail {
template <typename T>
constexpr bool kMaybeBase = std::is_class<T>::value && !std::is_final<T>::value;

// Subclass for storing the payload of the AsyncValue inline.  This should
/// never be directly accessed by users - always use AsyncValue methods instead.
class SomeConcreteAsyncValue : public AsyncValue {
  friend class AsyncValue;
  template <typename T>
  friend class ConcreteAsyncValue;
  using AsyncValue::AsyncValue;

  //===--------------------------------------------------------------------===//
  // TypeID and Destructor related functionality
  //===--------------------------------------------------------------------===//

  // We don't want a virtual function pointer in AsyncValue because it is too
  // big. Accordingly, we need another way to get a pointer to the destructor
  // an arbitrary type T.  To solve for this, we store the function pointers in
  // a side table and use 16-bit indexes into it.
public:
  // This is the signature for the destructor function for some value.
  using ValueDestructorFn = void (*)(void *);

  /// This function destroys a value of type T at the specified address.
  template <typename T>
  static void destructorFnPtr(void *pointer) {
    static_cast<T *>(pointer)->~T();
  }

private:
  // Only invoked by destroyWithRefCountZero.
  ~SomeConcreteAsyncValue();

  /// isTypeCompatible returns true if the type value stored in this AsyncValue
  /// instance can be safely cast to `T`. This is a conservative check:
  /// isTypeCompatible may return true even if the value cannot be safely cast
  /// to `T`. However, if it returns false then the value definitely cannot be
  /// safely cast to `T`. This means it is useful mainly as a debugging aid for
  /// use in assert() etc.
  template <typename T,
            typename std::enable_if<kMaybeBase<T>>::type * = nullptr>
  bool isTypeCompatible() const {
    // `T` might be a baseclass of the concrete type held by this AsyncValue.
    return true;
  }
  template <typename T,
            typename std::enable_if<!kMaybeBase<T>>::type * = nullptr>
  bool isTypeCompatible() const {
    return getTypeID<T>() == getTypeID();
  }

  /// Return the stored destructor function for this ConcreteValue.
  ValueDestructorFn getValueDestructor();

  /// The error value is always first thing in our derived class.
  EncodedDiagnostic *getDiagnosticPointer() {
    return reinterpret_cast<EncodedDiagnostic *>(this + 1);
  }

  /// The waiter value is always the firstthing in our derived class.
  Waiter *getWaiterPointer() { return reinterpret_cast<Waiter *>(this + 1); }

  /// Return the address of the (potentially uninitialized) payload.
  void *getPayloadPointer() {
    /// The payload in a ConcreteAsyncValue always immediately follows the
    /// AsyncValue.  This is guaranteed by static_asserts in ConcreteAsyncValue
    /// below.
    return this + 1;
  }

  /// This is the out-of-line slow patch for type registration.
  static void doTypeRegistration(std::atomic<uint16_t> *staticTypeID,
                                 ValueDestructorFn destructor);

  WaitersAndState removeAnyInlineWaiter(llvm::Optional<Waiter> &inlineWaiter,
                                        State newState);
};

/// Subclass for storing the payload of the AsyncValue inline.  This should
/// never be directly accessed by users - always use AsyncValue methods instead.
template <typename T>
class ConcreteAsyncValue : public SomeConcreteAsyncValue {
  friend class AsyncValue;
  friend class SomeConcreteAsyncValue;
  /// Allocate an instance of ConcreteAsyncValue in the specified state, but
  /// with the payload uninitialized.
  static ConcreteAsyncValue<T> *allocate(State state,
                                         CompactRuntimePtr runtime) {
    assert(ConcreteAsyncValue<T>::staticTypeID.load(
               std::memory_order_relaxed) != uint16_t(~0U) &&
           "AsyncValue type not registered");
    auto *ptr = (ConcreteAsyncValue<T> *)M::alignedAlloc(
        sizeof(ConcreteAsyncValue<T>), alignof(ConcreteAsyncValue<T>));
    new (ptr) ConcreteAsyncValue<T>(state, std::is_polymorphic_v<T>,
                                    getTypeID<T>(), runtime);
    return ptr;
  }

  /// Register our T type, setting `staticTypeID` to a non-sentinel value and
  /// remembering our destructor function in a side table.
  static void registerType() {
    if (ConcreteAsyncValue<T>::staticTypeID.load(std::memory_order_acquire) !=
        uint16_t(~0U))
      return;
    doTypeRegistration(&ConcreteAsyncValue<T>::staticTypeID,
                       SomeConcreteAsyncValue::destructorFnPtr<T>);
  }

private:
  ConcreteAsyncValue(State state, bool hasVTable, uint16_t typeID,
                     CompactRuntimePtr runtime)
      : SomeConcreteAsyncValue(SubclassKind::kConcrete, state, hasVTable,
                               typeID, runtime) {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
    static_assert(offsetof(ConcreteAsyncValue<T>, payload) ==
                      AsyncValue::kAsyncValueSize,
                  "Offset of ConcreteAsyncValue::payload needs to be aligned");
#pragma GCC diagnostic pop
#endif
  }

  // NOTE: destruction of this state is handled by ~SomeConcreteAsyncValue.
  union {
    /// When in a `kError` state, this includes location and diagnostic
    /// information.  Both this field and Waiter are 3 words.
    EncodedDiagnostic diagnostic;
    /// When unconstructed, this can hold an inline copy of the first waiter,
    /// avoiding having to heap allocate a waiter node for it.  The state will
    /// be either `kUnconstructedInlineWaiterConstructing` (while initializing
    /// this field for a brief few cycles) or
    /// `kUnconstructedInlineWaiterPresent` after initialization.
    Waiter waiter;
    /// When in `kConstructed` or `kAvailable` state, this is the payload of
    /// the AsyncValue.
    T payload;
  };

  // AsyncValue needs to be 16 bytes in size to assure that payloads in
  // ConcreteAsyncValue are 16-byte aligned.  This simplifies all clients.
  static_assert(sizeof(AsyncValue) == kAsyncValueSize,
                "Unexpected size for AsyncValue");

  static std::atomic<uint16_t> staticTypeID;
};

template <typename T>
std::atomic<uint16_t> ConcreteAsyncValue<T>::staticTypeID(uint16_t(~0));
} // end namespace Detail.

namespace Detail {
/// IndirectAsyncValue represents an uncomputed AsyncValue of unspecified type.
/// This is used when an AsyncValue must be returned, but the value it holds is
/// not ready and the producer of the value doesn't know what type it will
/// ultimately be, or whether it will be an error.
class IndirectAsyncValue : public AsyncValue {
  friend class AsyncValue;
  IndirectAsyncValue(CompactRuntimePtr runtime)
      : AsyncValue(SubclassKind::kIndirect, State::kUnconstructed,
                   /*hasVTable=*/false,
                   /*typeID=*/uint16_t(~0U), runtime) {}
  ~IndirectAsyncValue() = default;

  AnyAsyncValueRef value;
};
} // end namespace Detail

//===----------------------------------------------------------------------===//
// AsyncValue inline method implementations.
//===----------------------------------------------------------------------===//

/// Type registration - AsyncValue requires that each static type be
/// registered ahead of their use in an AsyncValue.  This method is efficient
/// in the case where a type is already registered, so it is fine to register
/// types without guarding against duplicates etc.
template <typename T>
void AsyncValue::registerType() {
  Detail::ConcreteAsyncValue<T>::registerType();
}

/// Helper function that calls registerType() for each type in the list.
template <typename... Ts>
void AsyncValue::registerTypes() {
  (AsyncValue::registerType<Ts>(), ...);
}

/// Create an AsyncValue for the specified type in "unconstructed" state.
template <typename T>
inline AnyAsyncValueRef AsyncValue::allocate(CompactRuntimePtr runtime) {
  return takeRCRef(
      Detail::ConcreteAsyncValue<T>::allocate(State::kUnconstructed, runtime));
}

/// Create an AsyncValue for the specified type in "constructed" but non-ready
/// state.  When the value is finalized, you should call `markReady()`, or
/// `setToError` to mark it as ready and notify waiters.
template <typename T, typename... Args>
inline AnyAsyncValueRef AsyncValue::createConstructed(CompactRuntimePtr runtime,
                                                      Args &&...args) {
  auto *result =
      Detail::ConcreteAsyncValue<T>::allocate(State::kConstructed, runtime);
  new (&result->payload) T(std::forward<Args>(args)...);
  return takeRCRef(result);
}

/// Create an AsyncValue for the specified type in "available" and ready state.
/// This is a terminal state for an AsyncValue, it can never change out of this
/// state.
template <typename T, typename... Args>
inline AnyAsyncValueRef AsyncValue::createReady(CompactRuntimePtr runtime,
                                                Args &&...args) {
  auto *result =
      Detail::ConcreteAsyncValue<T>::allocate(State::kAvailable, runtime);
  new (&result->payload) T(std::forward<Args>(args)...);
  return takeRCRef(result);
}

inline void AsyncValue::addRef() {
  assert(refcount.load() > 0);
  ++refcount;
}

inline void AsyncValue::addRef(uint16_t count) {
  if (count > 0) {
    assert(refcount.load() > 0);
    // Increasing the reference counter can always be done with
    // memory_order_relaxed: New references to an object can only be formed
    // from an existing reference, and passing an existing reference from one
    // thread to another must already provide any required synchronization.
    refcount.fetch_add(count, std::memory_order_relaxed);
  }
}

inline void AsyncValue::dropRef(uint16_t count) {
  assert(refcount.load() > 0);
  // We expect that `count` argument will often equal the actual reference count
  // here; optimize for that.  If `count` == reference count, only an acquire
  // barrier is needed to prevent the effects of the deletion from leaking
  // before this point.
  //
  // TODO: Measure and evaluate whether this is a useful optimization on all
  // systems.  On X86 systems for example, this is probably not actually a win.
  bool isLastRef = refcount.load(std::memory_order_acquire) == count;
  if (!isLastRef) {
    // If `count` != reference count, a release barrier is needed in
    // addition to an acquire barrier so that prior changes by this thread
    // cannot be seen to occur after this decrement.
    isLastRef = refcount.fetch_sub(count, std::memory_order_acq_rel) == count;
  }

  // Destroy this value if the refcount drops to zero.
  if (isLastRef)
    destroyWithRefCountZero();
}

/// Call the specified closure if the value is ready.  Otherwise, add it
/// to the waiter list and calls it when the value becomes ready.
template <typename WaiterT>
inline auto AsyncValue::andThen(WaiterT &&waiter)
    -> decltype(waiter(), void()) {
  andThen([waiter = std::forward<WaiterT>(waiter)](const AnyAsyncValueRef &) {
    return waiter();
  });
}

/// Call the specified closure if the value is ready.  Otherwise, add it
/// to the waiter list and calls it when the value becomes ready.  This
/// overload passes the current value back into the closure as a
/// `const AnyAsyncValueRef &`.  This eliminates the need to capture the
/// receiver in the closure and reduces reference count traffic.
template <typename WaiterT>
inline auto AsyncValue::andThen(WaiterT &&waiter)
    -> decltype(waiter(AnyAsyncValueRef()), void()) {
  // Clients generally want to use andThen without them each having to check
  // to see if the value is present. Check for them, and immediately run the
  // lambda if it is already here.
  auto waitersAndStateValue = waitersAndState.load(std::memory_order_acquire);
  if (isReady(waitersAndStateValue.getInt())) {
    assert(waitersAndStateValue.getPointer() == nullptr);
    runOneWaiter(waiter);
    return;
  }

  (void)andThenOutOfLine(std::forward<WaiterT>(waiter), waitersAndStateValue);
}

/// Construct the payload of a ConcreteAsyncValue and change its state to
/// `kConstructed`.  Requires that the AsyncValue's state is `kUnconstructed`,
/// and is moved to a ready state with `markReady()`.
template <typename T, typename... Args>
inline void AsyncValue::construct(Args &&...args) {
  assert(getTypeID<T>() == typeID && "Incorrect accessor");
  assert(getSubclassKind() == SubclassKind::kConcrete &&
         "cannot construct an IndirectAsyncvalue");
  auto *concrete = static_cast<Detail::ConcreteAsyncValue<T> *>(this);

  // Take any inline waiters out of the payload so we can construct T into it.
  llvm::Optional<Waiter> inlineWaiter;
  auto oldValue =
      concrete->removeAnyInlineWaiter(inlineWaiter, State::kConstructed);

  // If we had an inline waiter, re-add it onto the traditional waiter list.
  if (inlineWaiter.hasValue())
    oldValue = andThenOutOfLine(std::move(*inlineWaiter), oldValue);

  // We now have unfettered access to the payload section.
  new (&concrete->payload) T(std::forward<Args>(args)...);

  // Change the state to 'constructed' while making sure any waiters that
  // get concurrently added don't get lost.  We know that no other state
  // transition can happen concurrently.
  auto result = moveState(oldValue, State::kConstructed);
  assert(succeeded(result));
  (void)result;
}

/// Construct the payload of the AsyncValue in place and change its state to
/// kConcrete. Requires that this is a ConcreteAsyncValue that have state
/// `kUnconstructed`.
template <typename T, typename... Args>
inline void AsyncValue::emplace(Args &&...args) {
  assert(getSubclassKind() == SubclassKind::kConcrete &&
         "Cannot 'emplace' an IndirectValue, use 'emplaceIndirect' instead");
  assert(getTypeID<T>() == typeID && "Incorrect accessor");
  auto *concrete = static_cast<Detail::ConcreteAsyncValue<T> *>(this);

  // Take any inline waiters out of the payload area so we can construct it.
  llvm::Optional<Waiter> inlineWaiter;
  (void)concrete->removeAnyInlineWaiter(inlineWaiter, State::kAvailable);

  // Initialize the payload.
  new (&concrete->payload) T(std::forward<Args>(args)...);

  // Change state and notify the waiters.
  auto oldState = notifyReady(State::kAvailable, &inlineWaiter);
  assert((oldState == State::kUnconstructedInlineWaiterPresent ||
          oldState == State::kConstructed) &&
         "AsyncValue transitioned to ready while we're emplacing?");
  (void)oldState;
}

/// Construct the payload of the AsyncValue in place and change its state to
/// kConcrete. Requires that this is a ConcreteAsyncValue that have state
/// `kUnconstructed`.
template <typename T, typename... Args>
inline void AsyncValue::emplaceIndirect(Args &&...args) {
  assert(getSubclassKind() == SubclassKind::kIndirect);
  resolveIndirect(
      createReady<T, Args...>(getRuntime(), std::forward<Args>(args)...));
}

template <typename T>
const T &AsyncValue::get() const {
  assert(isConstructedOrAvailable(getState()) &&
         "Cannot call get() when AsyncValue isn't constructed");
  if (getSubclassKind() == SubclassKind::kConcrete) {
    auto *thisConcrete =
        static_cast<const Detail::ConcreteAsyncValue<T> *>(this);
    // Make sure both T (the stored type) and BaseT have a VTable or
    // neither have the VTable.
    assert(thisConcrete->template isTypeCompatible<T>() &&
           std::is_polymorphic_v<T> == hasVTable && "incorrect accessor");
    return thisConcrete->payload;
  }

  auto *thisIndirect = static_cast<const Detail::IndirectAsyncValue *>(this);
  assert(thisIndirect->value &&
         "indirect can't be constructed without being resolved");
  return thisIndirect->value->get<T>();
}

} // namespace LLCL

#endif // LLCL_RUNTIME_ASYNCVALUE_H
