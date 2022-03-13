//===- AsyncValue.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/ConcurrentAppendingVector.h"
using namespace LLCL;

std::atomic<ssize_t> AsyncValue::totalAllocatedAsyncValues{0};

//===----------------------------------------------------------------------===//
// TypeID and Destructor related functionality
//===----------------------------------------------------------------------===//

using DestructorFn = Detail::SomeConcreteAsyncValue::DestructorFn;

static ConcurrentAppendingVector<DestructorFn> &getTypeInfoTableSingleton() {
  static auto *table =
      new ConcurrentAppendingVector<DestructorFn>(/*initial capacity*/ 64);
  return *table;
}

auto Detail::SomeConcreteAsyncValue::getDestructor() -> DestructorFn {
  auto &table = getTypeInfoTableSingleton();
  uint16_t typeID = getTypeID();
  assert(typeID != 0);
  return table[getTypeID() - 1];
}

void Detail::SomeConcreteAsyncValue::doTypeRegistration(
    std::atomic<uint16_t> *staticTypeID, DestructorFn destructor) {
  size_t typeID = getTypeInfoTableSingleton().emplace_back(destructor) + 1;
  // Detect overflow.
  assert(typeID < std::numeric_limits<uint16_t>::max() &&
         "too many different AsyncValue types");

  // Set the value to the entry ID if we're the first one to do so.  If some
  // other thread beat us here, then we just abandon the table entry.  We don't
  // actually care if we succeed or if some other thread succeeded.
  uint16_t existing = uint16_t(~0U);
  (void)staticTypeID->compare_exchange_strong(existing, uint16_t(typeID),
                                              std::memory_order_release,
                                              std::memory_order_acquire);
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

void AsyncValue::destroyWithRefCountZero() {
  if (getSubclassKind() != SubclassKind::kIndirect) {
    auto *concrete = static_cast<Detail::SomeConcreteAsyncValue *>(this);
    concrete->getDestructor()(concrete);
    M::alignedFree(this);
    return;
  }

  delete static_cast<Detail::IndirectAsyncValue *>(this);
}

//===----------------------------------------------------------------------===//
// Waiter list management.
//===----------------------------------------------------------------------===//

namespace LLCL {
/// This is a singly linked list of nodes waiting for notification, hanging off
/// of AsyncValue.  When the AsyncValue becomes ready, the callbacks are
/// invoked.
class WaiterListNode {
public:
  explicit WaiterListNode(llvm::unique_function<void()> waiter)
      : waiterFn1(std::move(waiter)), nextAndIsFn2(nullptr, false) {}
  explicit WaiterListNode(
      llvm::unique_function<void(const RCRef<AsyncValue> &)> waiter)
      : waiterFn2(std::move(waiter)), nextAndIsFn2(nullptr, true) {}

  ~WaiterListNode() {
    if (hasClosureArgument())
      waiterFn2.~unique_function();
    else
      waiterFn1.~unique_function();
  }

  union {
    llvm::unique_function<void()> waiterFn1;
    llvm::unique_function<void(const RCRef<AsyncValue> &)> waiterFn2;
  };

  bool hasClosureArgument() const { return nextAndIsFn2.getInt(); }
  void setNext(WaiterListNode *newNext) { nextAndIsFn2.setPointer(newNext); }

  /// This is the next thing waiting on the AsyncValue, and a bit indicating
  /// whether we are waiterFn2 (true) or waiterFn1 (false).
  llvm::PointerIntPair<WaiterListNode *, 1, bool,
                       AsyncValue::WaiterListNodePointerTraits>
      nextAndIsFn2;

private:
  WaiterListNode(const WaiterListNode &) = delete;
  void operator=(const WaiterListNode &) = delete;
};
} // namespace LLCL

/// Invoke all of the waiters specified by the list of waiter nodes, and
/// deallocate the waiter nodes.
static void runWaitersAndDeallocate(AsyncValue *value, WaiterListNode *list) {
  // We pass the AsyncValue in as a `const RCRef<AsyncValue>&` to make the
  // ownership very clear (they can use the value but have to copy it if
  // persisting it).  We do this delicately to avoid additional refcount
  // bumps.
  auto rcThisRef = RCRef<AsyncValue>::take(value);

  while (list) {
    auto *node = list;
    if (node->hasClosureArgument())
      node->waiterFn2(rcThisRef);
    else
      node->waiterFn1();
    list = node->nextAndIsFn2.getPointer();
    delete node;
  }

  // We're done with the RCRef.
  (void)rcThisRef.release();
}

/// This is the out-of-line portion of the `AsyncValue::andThen` method which is
/// invoked when the value appears to be non-ready.
///
/// If the value is available or becomes available, this calls the closure
/// immediately. Otherwise, the add closure to the waiter list where it will be
/// called when the value becomes available.
void AsyncValue::andThenOutOfLine(WaiterListNode *node,
                                  WaitersAndState oldValue) {
  auto oldState = oldValue.getInt();

  // Swap the next link in. oldValue.getInt() must be non-ready when
  // evaluating the loop condition. The acquire barrier on the compare_exchange
  // ensures that prior changes to waiter list are visible here as we may call
  // RunWaiter() on it. The release barrier ensures that prior changes to *node
  // appear to happen before it's added to the list.
  node->setNext(oldValue.getPointer());
  auto newValue = WaitersAndState(node, oldState);
  while (!waitersAndState.compare_exchange_weak(oldValue, newValue,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
    // While swapping in our waiter, the value could have become ready.  If
    // so, just run the waiter and deallocate the node we don't need anymore.
    if (isReady(oldValue.getInt())) {
      assert(oldValue.getPointer() == nullptr);
      node->setNext(nullptr);
      runWaitersAndDeallocate(this, node);
      return;
    }
    // Otherwise, it is possible we just got extra waiter nodes.  Update the
    // waiter list in newValue.
    node->setNext(oldValue.getPointer());
  }

  // compare_exchange_weak succeeds. The oldValue must be in some non-ready
  // state.
  assert(!isReady(oldValue.getInt()));
}

/// This is the out-of-line portion of the `AsyncValue::andThen` method which is
/// invoked when the value appears to be non-ready.
///
/// If the value is available or becomes available, this calls the closure
/// immediately. Otherwise, the add closure to the waiter list where it will be
/// called when the value becomes available.
void AsyncValue::andThenOutOfLine(llvm::unique_function<void()> &&waiter,
                                  WaitersAndState oldValue) {
  // Create the node for our waiter.
  auto *node = new WaiterListNode(std::move(waiter));
  andThenOutOfLine(node, oldValue);
}

/// This is the out-of-line portion of the `AsyncValue::andThen` method which is
/// invoked when the value appears to be non-ready.
///
/// If the value is available or becomes available, this calls the closure
/// immediately. Otherwise, the add closure to the waiter list where it will be
/// called when the value becomes available.
void AsyncValue::andThenOutOfLine(
    llvm::unique_function<void(const RCRef<AsyncValue> &)> &&waiter,
    WaitersAndState oldValue) {
  // Create the node for our waiter.
  auto *node = new WaiterListNode(std::move(waiter));
  andThenOutOfLine(node, oldValue);
}

/// Transition to a ready state and notify all waiters about this.  This
/// returns the old state.
AsyncValue::State AsyncValue::notifyReady(State newState) {
  assert(newState == State::kAvailable || newState == State::kError);

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto oldValue = waitersAndState.exchange(WaitersAndState(nullptr, newState),
                                           std::memory_order_acq_rel);
  runWaitersAndDeallocate(this, oldValue.getPointer());
  return oldValue.getInt();
}

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

/// Create an AsyncValue that has already been turned into an error with the
/// specified message.
/// TODO: Add location support.
RCRef<AsyncValue> AsyncValue::createError(CompactRuntimePtr runtime,
                                          M::Error message) {
  auto *result =
      Detail::ConcreteAsyncValue<Chain>::allocate(State::kError, runtime);
  new (&result->error) M::Error(std::move(message));
  return takeRCRef(result);
}

/// Mark an "unconstructed" AsyncValue as an error.
/// TODO: Add location support.
void AsyncValue::setToError(M::Error message) {
  if (getSubclassKind() == SubclassKind::kConcrete) {
    assert(getState() == State::kUnconstructed &&
           "cannot set an error to an indirect or already set up AsyncValue");

    // We don't have the <T> type required to cast to ConcreteAsyncValue<T> so
    // do the pointer arithmetic manually.
    auto *errorPtr =
        static_cast<Detail::SomeConcreteAsyncValue *>(this)->getErrorPointer();
    new (errorPtr) M::Error(std::move(message));
    auto oldState = notifyReady(State::kError);
    assert(oldState == State::kUnconstructed &&
           "setting an erro to an AsyncValue that was already set up?");
    (void)oldState;
  } else {
    resolveIndirect(createError(getRuntime(), std::move(message)));
  }
}

//===----------------------------------------------------------------------===//
// IndirectAsyncValue implementation logic
//===----------------------------------------------------------------------===//

/// Resolve an IndirectAsyncValue to point to the specified new value,
/// resolving any waiters whenever newValue becomes ready.
void AsyncValue::resolveIndirect(RCRef<AsyncValue> newValue) {
  assert(getSubclassKind() == SubclassKind::kIndirect &&
         getState() == State::kUnconstructed &&
         "Can only resolve indirect async values");
  auto *thisIndirect = static_cast<Detail::IndirectAsyncValue *>(this);

  assert(!thisIndirect->value && "IndirectAsyncValue is already resolved");

  // If the newValue is already itself ready, we can resolve this indirect value
  // and make it ready.
  auto newValueState = newValue->getState();
  if (isReady(newValueState)) {
    // Collapse through an intermediate IndirectAsyncValue so they can be
    // deallocated and to reduce pointer hops.  We know there can be at most one
    // IndirectAsyncValue here because each locally resolves when they become
    // ready.
    if (newValue->getSubclassKind() == SubclassKind::kIndirect) {
      auto *concreteValue = newValue.getPointer();
      concreteValue = static_cast<Detail::IndirectAsyncValue *>(concreteValue)
                          ->value.getPointer();
      assert(concreteValue->getSubclassKind() == SubclassKind::kConcrete);
      newValue = copyRCRef(concreteValue);
    }

    // Resolve the type of the contained value.
    typeID = newValue->typeID;
    thisIndirect->value = std::move(newValue);

    // Finally, notify our waiters and switch to kAvailable or kError state.
    auto oldState = notifyReady(newValueState);
    assert(oldState == State::kUnconstructed &&
           "setting an erro to an AsyncValue that was already set up?");
    (void)oldState;
    return;
  }

  // Otherwise, the new value is still unresolved.  That's ok, we'll just wait
  // until it becomes ready and then try again.
  newValue->andThen(
      [this2 = copyRCRef(this)](const RCRef<AsyncValue> &newValue) mutable {
        this2->resolveIndirect(newValue.copy());
      });
}

/// Create an IndirectAsyncValue that may be filled in with any AsyncValue in
/// the future.
RCRef<AsyncValue> AsyncValue::createIndirect(CompactRuntimePtr runtime) {
  return takeRCRef(new Detail::IndirectAsyncValue(runtime));
}
