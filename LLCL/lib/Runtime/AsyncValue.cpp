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

using ValueDestructorFn = Detail::SomeConcreteAsyncValue::ValueDestructorFn;

static ConcurrentAppendingVector<ValueDestructorFn> &
getTypeInfoTableSingleton() {
  static auto *table =
      new ConcurrentAppendingVector<ValueDestructorFn>(/*initial capacity*/ 64);
  return *table;
}

auto Detail::SomeConcreteAsyncValue::getValueDestructor() -> ValueDestructorFn {
  auto &table = getTypeInfoTableSingleton();
  uint16_t typeID = getTypeID();
  assert(typeID != 0);
  return table[typeID - 1];
}

void Detail::SomeConcreteAsyncValue::doTypeRegistration(
    std::atomic<uint16_t> *staticTypeID, ValueDestructorFn destructor) {
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
// Destruction logic
//===----------------------------------------------------------------------===//

Detail::SomeConcreteAsyncValue::~SomeConcreteAsyncValue() {
  auto s = getState();
  // Destroy the error or value if constructed.
  if (s == State::kError)
    getDiagnosticPointer()->~EncodedDiagnostic();
  else if (s == State::kAvailable)
    getValueDestructor()(getPayloadPointer());
  else {
    // TODO: If unconstructed this will leak the waiters list.  We should signal
    // this as an error (checking for ressurection) etc.
    llvm::report_fatal_error(
        "destroying a non-available AsyncValue isn't implemented");
  }
}

void AsyncValue::destroyWithRefCountZero() {
  if (getSubclassKind() == SubclassKind::kConcrete) {
    auto *concrete = static_cast<Detail::SomeConcreteAsyncValue *>(this);
    concrete->~SomeConcreteAsyncValue();
    M::alignedFree(this);
    return;
  }

  delete static_cast<Detail::IndirectAsyncValue *>(this);
}

//===----------------------------------------------------------------------===//
// Waiter list management.
//===----------------------------------------------------------------------===//

/// Prepare to transition a ConcreteAsyncValue to a state specified by
/// `newState`, which must be either `kAvailable` or `kError`.
/// The origin state may be `kUnconstructed` or `kUnconstructedInlineWaiter*`.
///
/// The postcondition of this method is that the waiter state of the async value
/// is guaranteed to be in a state that doesn't allow adding inline waiters, but
/// any inline waiters will be moved out to the `inlineWaiter` argument, and
/// the payload/error area is uninitialized.
///
void AsyncValue::removeAnyInlineWaiter(llvm::Optional<Waiter> &inlineWaiter) {
  WaitersAndState oldValue = waitersAndState.load(std::memory_order_acquire);
  while (1) { // This loop allows us to 'continue' to retry or `return` to exit.
    switch (oldValue.getInt()) {
    default:
      assert(0 && "cannot construct a ready AsyncValue");
    case State::kUnconstructed: {
      assert(oldValue.getPointer() == nullptr &&
             "how'd we get out of line waiters without an inline waiter?");
      // We need to avoid races with other threads "andThen'ing" the async value
      // which would try to set up an inline waiter.  We do this by moving our
      // state to kUnconstructedInlineWaiterPresent because any andThen would
      // put the waiter on the waiter list if we get to that state.  We have to
      // be careful though because we might not successfully get to that state!
      auto newValue =
          WaitersAndState(nullptr, State::kUnconstructedInlineWaiterPresent);
      if (!waitersAndState.compare_exchange_weak(oldValue, newValue,
                                                 std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
        // If we failed the compare/xchg, retry.  The only state transition is
        // to having an inline waiter and potentially indirect waiters.
        continue;
      }
      // When we succeed, we're in a 'inline waiter present' state, but
      // the inline waiter isn't actually constructed.
      return;
    }
    case State::kUnconstructedInlineWaiterConstructing:
      // If someone is actively constructing a waiter, spin for a few cycles
      // until it resolves.
      oldValue = waitersAndState.load(std::memory_order_acquire);
      // TODO: We should sleep 10 cycles and potentially even give up the thread
      // or something, how do we do this?  ThreadPoolWorkQueue uses
      // std::chrono::steady_clock::now().
      // We should potentially do exponential backoff in all these compare/xchg
      // loops.
      continue;

    case State::kUnconstructedInlineWaiterPresent: {
      Waiter *waiterPtr = getWaiterPointer();
      // If we have an inline waiter, move it aside.
      inlineWaiter = std::move(*waiterPtr);
      waiterPtr->~Waiter();
      return;
    }
    }
  }
}

namespace LLCL {
class WaiterListNode {
public:
  explicit WaiterListNode(AsyncValue::Waiter &&newWaiter, WaiterListNode *next)
      : waiter(std::move(newWaiter)), next(next) {}
  friend class AsyncValue;

private:
  AsyncValue::Waiter waiter;
  WaiterListNode *next = nullptr;

  WaiterListNode(const WaiterListNode &) = delete;
  void operator=(const WaiterListNode &) = delete;
};
} // namespace LLCL

/// Invoke all of the waiters specified by the list of waiter nodes, and
/// deallocate the waiter nodes.  We know we have ownership of `list` here and
/// that it is done being mutated.  We also know that the caller has an RCRef
/// that keeps 'this' alive.
void AsyncValue::runWaitersAndDeallocate(WaiterListNode *list) {
  while (list) {
    auto *node = list;
    runOneWaiter(node->waiter);
    list = node->next;
    delete node;
  }
}

/// Atomically move from the current state specified by `oldValue` to the
/// state specified by `newState`, ignoring any waiter changes.  This returns
/// success() when successful, or failure() if the AsyncValue moved to another
/// state in the meantime.
M::LogicalResult AsyncValue::moveState(WaitersAndState &oldValue,
                                       State newState) {
  auto origState = oldValue.getInt();
  assert(origState != newState && "cannot transition to same state");
  auto newValue = WaitersAndState(oldValue.getPointer(), newState);
  while (!waitersAndState.compare_exchange_weak(oldValue, newValue,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
    // If the thing changed to a different state underneath us, return
    // failure.
    if (oldValue.getInt() != origState)
      return M::failure();

    // If the waiter list changed out from under us, try again.
    newValue = WaitersAndState(oldValue.getPointer(), newState);
  }
  oldValue = newValue;
  return M::success();
}

/// This is the out-of-line portion of the `AsyncValue::andThen` method which is
/// invoked when the value appears to be non-ready.
///
/// If the value is available or becomes available, this calls the closure
/// immediately. Otherwise, the add the waiter closure to the waiter list where
/// it will be called when the value becomes available.
void AsyncValue::andThenOutOfLine(Waiter waiter, WaitersAndState oldValue) {
  // If the oldValue appears to be kUnconstructed then we will try to add this
  // as an inline waiter node, otherwise we add another node to the waiter list.
  if (oldValue.getInt() == State::kUnconstructed) {
    assert(oldValue.getPointer() == nullptr &&
           "how'd we get out of line waiters without an inline waiter?");
    // Allocate the payload aread by moving to the ...Constructing state.  If
    // the AsyncValue moved to another state in the meantime then it either
    // got constructed or became available.  In any case, we can't do inline
    // waiter initialization so we have to fall back.
    if (succeeded(moveState(oldValue,
                            State::kUnconstructedInlineWaiterConstructing))) {
      // In the vastly most common case we get into the 'WaiterConstructing'
      // state. Inline initialize the waiter.
      new (getWaiterPointer()) Waiter(std::move(waiter));
      // Then transition immediately to the 'WaiterPresent' state.  We want this
      // critical section to be extremely short, only a few cycles.
      auto result =
          moveState(oldValue, State::kUnconstructedInlineWaiterPresent);
      assert(succeeded(result) &&
             "no state transitions can happen in this window");
      (void)result;
      return;
    }
  }

  // If we raced with a transition into a ready state then we can just execute
  // the waiter and be done.
  if (isReady(oldValue.getInt()))
    return runOneWaiter(waiter);

  // Otherwise, the value is inline waiter is occupied and the state is
  // unavailable, go ahead and do some head allocations.
  auto node = new WaiterListNode(std::move(waiter), oldValue.getPointer());

  // Swap the next link in. oldValue.getInt() must be non-ready when
  // evaluating the loop condition. The acquire barrier on the
  // compare_exchange ensures that prior changes to waiter list are visible
  // here as we may call RunWaiter() on it. The release barrier ensures that
  // prior changes to *node appear to happen before it's added to the list.
  auto newValue = WaitersAndState(node, oldValue.getInt());
  while (!waitersAndState.compare_exchange_weak(oldValue, newValue,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
    // While swapping in our waiter, the value could have become ready.  If
    // so, just run the waiter and deallocate the node we don't need anymore.
    if (isReady(oldValue.getInt())) {
      assert(oldValue.getPointer() == nullptr);
      runOneWaiter(node->waiter);
      // Change the tail of the list to null.  Whatever moved this to a ready
      // state will already have executed and deallocated that.
      node->next = nullptr;
      delete node;
      return;
    }
    // Otherwise, it is possible we just got extra waiter nodes.  Update the
    // waiter list in newValue.
    node->next = oldValue.getPointer();
  }

  // compare_exchange_weak succeeds. The oldValue must be in some non-ready
  // state.
  assert(!isReady(oldValue.getInt()));
}

/// Transition to a ready state and notify all waiters about this.  This
/// returns the old state.
AsyncValue::State AsyncValue::notifyReady(State newState,
                                          llvm::Optional<Waiter> &extraWaiter) {
  assert((newState == State::kAvailable || newState == State::kError) &&
         "new state isn't a ready state!");

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto oldValue = waitersAndState.exchange(WaitersAndState(nullptr, newState),
                                           std::memory_order_acq_rel);

  // If there was an inline waiter, run it first.
  if (extraWaiter.hasValue())
    runOneWaiter(*extraWaiter);

  //  Then run the rest of the waiter list.
  runWaitersAndDeallocate(oldValue.getPointer());
  return oldValue.getInt();
}

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

/// If this AsyncValue holds an error, return its diagnostic.  If not, return
/// nullptr.
EncodedDiagnostic *AsyncValue::getDiagnosticIfPresent() {
  // If this isn't an error, we're done.
  if (getState() != State::kError)
    return nullptr;

  if (getSubclassKind() == SubclassKind::kConcrete)
    return static_cast<Detail::SomeConcreteAsyncValue *>(this)
        ->getDiagnosticPointer();

  auto *thisIndirect = static_cast<Detail::IndirectAsyncValue *>(this);
  return thisIndirect->value->getDiagnosticIfPresent();
}

/// Create an AsyncValue that has already been turned into an error with the
/// specified message.
AnyAsyncValueRef AsyncValue::createError(CompactRuntimePtr runtime,
                                         EncodedDiagnostic diagnostic) {
  auto *result =
      Detail::ConcreteAsyncValue<Chain>::allocate(State::kError, runtime);
  new (&result->diagnostic) EncodedDiagnostic(std::move(diagnostic));
  return takeRCRef(result);
}

/// Mark an "unconstructed" AsyncValue as an error.
void AsyncValue::setToError(EncodedDiagnostic diagnostic) {
  if (getSubclassKind() != SubclassKind::kConcrete) {
    resolveIndirect(createError(getRuntime(), std::move(diagnostic)));
    return;
  }

  llvm::Optional<Waiter> inlineWaiter;
  removeAnyInlineWaiter(inlineWaiter);

  auto *concrete = static_cast<Detail::SomeConcreteAsyncValue *>(this);
  auto *diagPtr = concrete->getDiagnosticPointer();
  new (diagPtr) EncodedDiagnostic(std::move(diagnostic));
  auto oldState = notifyReady(State::kError, inlineWaiter);
  assert(oldState == State::kUnconstructedInlineWaiterPresent &&
         "AsyncValue transitioned to ready while we're changing to error?");
  (void)oldState;
}

//===----------------------------------------------------------------------===//
// IndirectAsyncValue implementation logic
//===----------------------------------------------------------------------===//

/// Resolve an IndirectAsyncValue to point to the specified new value,
/// resolving any waiters whenever newValue becomes ready.
void AsyncValue::resolveIndirect(AnyAsyncValueRef newValue) {
  assert(getSubclassKind() == SubclassKind::kIndirect && !isReady(getState()) &&
         "Can only resolve indirect async values");
  auto *thisIndirect = static_cast<Detail::IndirectAsyncValue *>(this);

  // If the newValue is already itself ready, we can resolve this indirect
  // value and make it ready.
  auto newValueState = newValue->getState();
  if (isReady(newValueState)) {
    // Collapse through an intermediate IndirectAsyncValue so they can be
    // deallocated and to reduce pointer hops.  We know there can be at most
    // one IndirectAsyncValue here because each locally resolves when they
    // become ready.
    if (newValue->getSubclassKind() == SubclassKind::kIndirect) {
      auto *concreteValue = newValue.getPointer();
      concreteValue = static_cast<Detail::IndirectAsyncValue *>(concreteValue)
                          ->value.getPointer();
      assert(concreteValue->getSubclassKind() == SubclassKind::kConcrete);
      newValue = copyRCRef(concreteValue);
    }

    // Resolve the type of the contained value.
    typeID = newValue->typeID;

    llvm::Optional<Waiter> inlineWaiter;
    removeAnyInlineWaiter(inlineWaiter);

    new (&thisIndirect->value) AnyAsyncValueRef(std::move(newValue));

    // Finally, notify our waiters and switch to kAvailable or kError state.
    auto oldState = notifyReady(newValueState, inlineWaiter);
    assert(!isReady(oldState) &&
           "resolving an IndirectAsyncValue that was already set up?");
    (void)oldState;
    return;
  }

  // Otherwise, the new value is still unresolved.  That's ok, we'll just wait
  // until it becomes ready and then try again.
  newValue->andThen(
      [this2 = copyRCRef(this)](const AnyAsyncValueRef &newValue) mutable {
        this2->resolveIndirect(newValue.copy());
      });
}

/// Create an IndirectAsyncValue that may be filled in with any AsyncValue in
/// the future.
AnyAsyncValueRef AsyncValue::createIndirect(CompactRuntimePtr runtime) {
  return takeRCRef(new Detail::IndirectAsyncValue(runtime));
}
