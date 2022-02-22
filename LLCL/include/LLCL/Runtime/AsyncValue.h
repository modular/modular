//===- LLCL/Runtime/AsyncValue.h --------------------------------*- C++ -*-===//
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
#include "Support/AlignedAlloc.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/PointerIntPair.h"

namespace LLCL {
class Runtime;
class WaiterListNode;
namespace Detail {
template <typename T>
class ConcreteAsyncValue;
}

/// This is a future of the specified value type. Arbitrary C++ types may be
/// used here, even non-copyable types and expensive ones like tensors.  All
/// AsyncValues are allocated out of a specific `Runtime` instance and can
/// identify them with `getRuntime()`.
///
/// An AsyncValue is in one of four states (unconstructed, constructed,
/// available, error), where the first two are considered "non-ready" and the
/// last two are considered "ready" (waiters are notified).   If it is in the
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
  //===--------------------------------------------------------------------===//
  // Static creation methods for AsyncValue's
  //===--------------------------------------------------------------------===//

  /// Create an AsyncValue for the specified type in "unconstructed" state.
  /// This should be `emplace`'d, `construct`'d, or finalized with an error.
  template <typename T>
  static AsyncValue *createUnconstructed(CompactRuntimePtr runtime);

  /// Create an AsyncValue for the specified type in "constructed" but non-ready
  /// state.  When This should be `markReady()`, or finalized with an error.
  template <typename T, typename... Args>
  static AsyncValue *createConstructed(CompactRuntimePtr runtime,
                                       Args &&...args);

  /// Create an AsyncValue for the specified type in "available" and ready
  /// state. This is a terminal state for an AsyncValue, it can never change out
  /// of this state.
  template <typename T, typename... Args>
  static AsyncValue *createReady(CompactRuntimePtr runtime, Args &&...args);

  //===--------------------------------------------------------------------===//
  // State change methods.
  //===--------------------------------------------------------------------===//

  /// Construct the payload of a ConcreteAsyncValue and change our state to
  /// `kAvailable`.  Requires that this AsyncValue's state is `unconstructed`.
  template <typename T, typename... Args>
  void emplace(Args &&...args);

  /// Transition a "constructed" AsyncValue to "available" and notify any
  /// waiters.
  void markReady() {
    auto oldState = notifyReady(State::kAvailable);
    assert(oldState == State::kConstructed &&
           "can only mark 'constructed' values ready");
    (void)oldState;
  }

  //===--------------------------------------------------------------------===//
  // Primary interface to AsyncValue for clients to use.
  //===--------------------------------------------------------------------===//

  /// Call the specified closure if the value is ready.  Otherwise, add it
  /// to the waiter list and calls it when the value becomes ready.
  template <typename WaiterT>
  void andThen(WaiterT &&waiter);

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
  /// The above conditions are required since we store only the offset of the
  /// payload type in AsyncValue as data_traits_.buf_offset. Violation of either
  /// 1) or 2) requires additional pointer adjustments to get the proper pointer
  /// for the base type, which we do not have sufficient information to perform
  /// at runtime.
  template <typename T>
  const T &get() const;

  // Same as the const overload of get(), for mutable use-cases.
  template <typename T>
  T &get() {
    return const_cast<T &>(static_cast<const AsyncValue *>(this)->get<T>());
  }

  // TODO: Handle Errors.

  /// Return the `Runtime` instance this is part of.
  CompactRuntimePtr getRuntime() const { return runtime; }

  //===--------------------------------------------------------------------===//
  // Type Related functionality
  //===--------------------------------------------------------------------===//

  /// Return a type identifier for the payload held by this AsyncValue.  In the
  /// case of an IndirectAsyncValue, this will be meaningless.
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
    /// The underlying value's constructor has not been invoked and the value is
    /// not ready for consumption. This state can transition to `kConstructed`,
    /// `kAvailable` and `kError`.
    kUnconstructed = 0,

    /// The underlying value's constructor is called but the value is not
    /// ready for consumption (triggering waiters). This state can
    /// transition to `available` and `error`.
    kConstructed = 1,

    /// The underlying value is constructed and ready for consumption by
    /// waiters and contains an initialized value. This state can not transition
    /// to any other state.
    kAvailable = 2,

    /// This AsyncValue is ready and contains an error, along with an
    /// uninitialized value. This state can not transition to any other state.
    kError = 3,
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

  /// Increase the reference count.
  void addRef();
  void addRef(uint16_t count);

  /// Decrease the reference count of this object, potentially deallocating it.
  void dropRef(uint16_t count = 1);

private:
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

  // hasVTable has the same value for a given payload type T.
  const bool hasVTable : 1;

  // NOTE: 6 unused padding bits.

  // This is a 16-bit value that identifies the type.
  const uint16_t typeID;

  struct WaiterListNodePointerTraits {
    static inline void *getAsVoidPointer(WaiterListNode *ptr) { return ptr; }
    static inline WaiterListNode *getFromVoidPointer(void *ptr) {
      return static_cast<WaiterListNode *>(ptr);
    }
    enum { NumLowBitsAvailable = 2 };
  };

  /// The waiter list and the state are compacted into a single atomic word,
  /// since the fields need to be accessed at the same time for state changes.
  ///
  /// Invariant: If the state is ready, then the waiter list must be nullptr.
  using WaitersAndState = llvm::PointerIntPair<WaiterListNode *, 2, State,
                                               WaiterListNodePointerTraits>;
  std::atomic<WaitersAndState> waitersAndState;

protected:
  void andThenOutOfLine(llvm::unique_function<void()> &&waiter,
                        WaitersAndState oldValue);
  void destroyWithRefCountZero();

  /// Transition to a ready state and notify all waiters about this.  This
  /// returns the old state.
  State notifyReady(State newState);

protected:
  /// This layout of this class is designed very carefully to ensure alignment
  /// of the payload to 16 bytes.  That said, we do include a significant amount
  /// of metadata (including information about the concrete type, whether
  /// vtables exist or not, etc) in order to detect common programmer mistakes
  /// quickly.
  static constexpr int kAsyncValueSize = 16;

  AsyncValue(SubclassKind subclassKind, State state, bool hasVTable,
             uint16_t typeID, CompactRuntimePtr runtime)
      : runtime(runtime), subclassKind(subclassKind), hasVTable(hasVTable),
        typeID(typeID), waitersAndState(WaitersAndState(nullptr, state)) {}

  AsyncValue(const AsyncValue &) = delete;
  void operator=(const AsyncValue &) = delete;
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
  using AsyncValue::AsyncValue;

  //===--------------------------------------------------------------------===//
  // TypeID and Destructor related functionality
  //===--------------------------------------------------------------------===//

  // We don't want a virtual function pointer in AsyncValue because it is too
  // big. Accordingly, we need another way to get a pointer to the destructor
  // for ConcreteAsyncValue<T> whose details depend on the destructor for T.
  // To solve for this, we store the function pointers in a side table and
  // use 16-bit indexes into it.

public:
  // This is the signature for the destructor function.
  using DestructorFn = void (*)(AsyncValue *);

private:
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

  /// Return the stored destructor for this ConcreteValue.
  DestructorFn getDestructor();

  static uint16_t createTypeInfoAndReturnTypeIDImpl(DestructorFn destructor);

  // Creates a DestructorFn entry for `T` and store it in a global
  // TypeInfo table. Returns the "type id" for `T` which currently happens to
  // be one plus the index of this TypeInfo object in the TypeInfo table.
  //
  // This is only be called from the static initializer for the
  // ConcreteAsyncValue::staticTypeID field.
  template <typename T>
  static uint16_t createTypeInfoAndReturnTypeID() {
    return createTypeInfoAndReturnTypeIDImpl(
        Detail::ConcreteAsyncValue<T>::destructorFnPtr);
  }
};

/// Subclass for storing the payload of the AsyncValue inline.  This should
/// never be directly accessed by users - always use AsyncValue methods instead.
template <typename T>
class ConcreteAsyncValue : public SomeConcreteAsyncValue {
  friend class AsyncValue;
  friend class SomeConcreteAsyncValue;
  ~ConcreteAsyncValue() {
    static_assert(offsetof(ConcreteAsyncValue<T>, payload) ==
                      AsyncValue::kAsyncValueSize,
                  "Offset of ConcreteAsyncValue::payload needs to be aligned");

    auto s = getState();
    if (s == State::kError) {
      assert(0 && "errors not implemented yet");
      // delete error;
    } else if (isConstructedOrAvailable(s)) {
      payload.~T();
    }
  }

  // The destructor function for a ConcreteAsyncValue<T>.
  static void destructorFnPtr(AsyncValue *v) {
    static_cast<ConcreteAsyncValue<T> *>(v)->~ConcreteAsyncValue();
  }

  /// Allocate an instance of ConcreteAsyncValue in the specified state, but
  /// with the payload uninitialized.
  static ConcreteAsyncValue<T> *allocate(State state,
                                         CompactRuntimePtr runtime) {
    auto *ptr = (ConcreteAsyncValue<T> *)M::alignedAlloc(
        sizeof(ConcreteAsyncValue<T>), alignof(ConcreteAsyncValue<T>));
    new (ptr) ConcreteAsyncValue<T>(state, std::is_polymorphic_v<T>,
                                    getTypeID<T>(), runtime);
    return ptr;
  }

private:
  ConcreteAsyncValue(State state, bool hasVTable, uint16_t typeID,
                     CompactRuntimePtr runtime)
      : SomeConcreteAsyncValue(SubclassKind::kConcrete, state, hasVTable,
                               typeID, runtime) {}

  union {
    // TODO: DecodedDiagnostic *error;
    T payload;
  };

  // AsyncValue needs to be 16 bytes in size to assure that payloads in
  // ConcreteAsyncValue are 16-byte aligned.  This simplifies all clients.
  static_assert(sizeof(AsyncValue) == kAsyncValueSize,
                "Unexpected size for AsyncValue");

  static const uint16_t staticTypeID;
};

template <typename T>
const uint16_t ConcreteAsyncValue<T>::staticTypeID =
    ConcreteAsyncValue::createTypeInfoAndReturnTypeID<T>();
} // end namespace Detail.

//===----------------------------------------------------------------------===//
// AsyncValue inline method implementations.
//===----------------------------------------------------------------------===//

/// Create an AsyncValue for the specified type in "unconstructed" state.
template <typename T>
inline AsyncValue *AsyncValue::createUnconstructed(CompactRuntimePtr runtime) {
  return Detail::ConcreteAsyncValue<T>::allocate(State::kUnconstructed,
                                                 runtime);
}

/// Create an AsyncValue for the specified type in "constructed" but non-ready
/// state.  When This should be `markReady()`, or finalized with an error.
template <typename T, typename... Args>
inline AsyncValue *AsyncValue::createConstructed(CompactRuntimePtr runtime,
                                                 Args &&...args) {
  auto *result =
      Detail::ConcreteAsyncValue<T>::allocate(State::kConstructed, runtime);
  new (&result->payload) T(std::forward<Args>(args)...);
  return result;
}

/// Create an AsyncValue for the specified type in "available" and ready state.
/// This is a terminal state for an AsyncValue, it can never change out of this
/// state.
template <typename T, typename... Args>
inline AsyncValue *AsyncValue::createReady(CompactRuntimePtr runtime,
                                           Args &&...args) {
  auto *result =
      Detail::ConcreteAsyncValue<T>::allocate(State::kAvailable, runtime);
  new (&result->payload) T(std::forward<Args>(args)...);
  return result;
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

template <typename WaiterT>
inline void AsyncValue::andThen(WaiterT &&waiter) {
  // Clients generally want to use andThen without them each having to check
  // to see if the value is present. Check for them, and immediately run the
  // lambda if it is already here.
  auto waitersAndStateValue = waitersAndState.load(std::memory_order_acquire);
  if (isReady(waitersAndStateValue.getInt())) {
    assert(waitersAndStateValue.getPointer() == nullptr);
    waiter();
    return;
  }
  andThenOutOfLine(std::forward<WaiterT>(waiter), waitersAndStateValue);
}

/// Construct the payload of the AsyncValue in place and change its state to
/// kConcrete. Requires that this is a ConcreteAsyncValue that have state
/// `unavailable`.
template <typename T, typename... Args>
inline void AsyncValue::emplace(Args &&...args) {
  assert(getSubclassKind() == SubclassKind::kConcrete &&
         getState() == State::kUnconstructed &&
         "cannot emplace an indirect or already set up AsyncValue");
  assert(getTypeID<T>() == typeID && "Incorrect accessor");

  auto *concrete = static_cast<Detail::ConcreteAsyncValue<T> *>(this);
  new (&concrete->payload) T(std::forward<Args>(args)...);
  auto oldState = notifyReady(State::kAvailable);
  assert(oldState == State::kUnconstructed &&
         "fulfilling a concrete value that was already set up?");
  (void)oldState;
}

template <typename T>
const T &AsyncValue::get() const {
  if (getSubclassKind() == SubclassKind::kConcrete) {
    assert(isConstructedOrAvailable(getState()) &&
           "Cannot call get() when ConcreteAsyncValue isn't constructed");
    auto *thisConcrete =
        static_cast<const Detail::ConcreteAsyncValue<T> *>(this);
    // Make sure both T (the stored type) and BaseT have a VTable or
    // neither have the VTable.
    assert(thisConcrete->template isTypeCompatible<T>() &&
           std::is_polymorphic_v<T> == hasVTable && "incorrect accessor");
    return thisConcrete->payload;
  }

  assert(0 && "indirect not implemented yet");
  abort();
}

} // namespace LLCL

#endif // LLCL_RUNTIME_ASYNCVALUE_H
