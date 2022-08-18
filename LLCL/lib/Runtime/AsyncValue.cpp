//===- AsyncValue.cpp -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/Chain.h"
#include "LLCL/Support/ConcurrentAppendingVector.h"
#include "LLCL/Support/SpinWaiter.h"
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
  WaitersAndState oldValue = loadWaitersAndState();
  SpinWaiter<> spinWaiter;
  while (1) { // This loop allows us to 'continue' to retry or `return` to exit.
    switch (oldValue.getInt()) {
    case State::kAvailable:
    case State::kError:
      assert(0 && "cannot construct a ready AsyncValue");
    case State::kUnconstructed: {
      assert(oldValue.getPointer() == nullptr &&
             "how'd we get out of line waiters without an inline waiter?");
      // We need to avoid races with other threads "andThen'ing" the async value
      // which would try to set up an inline waiter.  We do this by moving our
      // state to kUnconstructed4ValidOOLWaiterSlots because any andThen
      // would put the waiter on the waiter list if we get to that state.  We
      // have to be careful though because we might not successfully get to that
      // state!
      auto newValue =
          WaitersAndState(nullptr, State::kUnconstructed4ValidOOLWaiterSlots);
      if (!compareExchangeWaiterAndState(oldValue, newValue)) {
        // If we failed the compare/xchg, retry.  The only state transition is
        // to having an inline waiter and potentially indirect waiters.
        if (spinWaiter.wait())
          oldValue = loadWaitersAndState();
        continue;
      }
      // When we succeed, we're in an 'inline waiter present' state, but
      // the inline waiter isn't actually constructed, so we don't put anything
      // into `inlineWaiter`.
      return;
    }
    case State::kUnconstructedInitializingInlineWaiter:
      // If someone is actively constructing a waiter, spin for a few cycles
      // until it resolves.
      spinWaiter.wait();
      oldValue = loadWaitersAndState();
      continue;

    case State::kUnconstructed1ValidOOLWaiterSlots:
    case State::kUnconstructed2ValidOOLWaiterSlots:
    case State::kUnconstructed3ValidOOLWaiterSlots:
    case State::kUnconstructed4ValidOOLWaiterSlots: {
      Waiter *waiterPtr = getInlineWaiterPointer();
      // If we have an inline waiter, move it aside.
      inlineWaiter = std::move(*waiterPtr);
      waiterPtr->~Waiter();
      return;
    }
    }
  }
}

namespace LLCL {
/// This class provides a singly linked list of nodes that each contain four
/// waiters.  The AsyncValue itself stores the first waiter added to an
/// AsyncValue inline in the same space as its payload field, then stores
/// additional waiters in this list.
///
/// Each node of this list holds four waiters - this reduces malloc overhead and
/// improves locality.  The thing that points to this node (the AV or another
/// list node) knows how many of the waiters are valid in this node.  We track
/// an atomic bitset in this node that keeps track of whether each entry in the
/// waiter array is fully initialized or not.
class WaiterListNode {
public:
  friend class AsyncValue;
  using Waiter = AsyncValue::Waiter;

  // Create a node with the first element initialized.
  explicit WaiterListNode(Waiter &&newWaiter, WaiterListNode *next)
      : firstWaiter(std::move(newWaiter)), next(next),
        waitersCompletelyInitialized(0) {}
  ~WaiterListNode() {
    // We know all the waiters will be destroyed at this point.
  }

  void setWaiter(size_t i, Waiter &&waiter) {
    // This slot shouldn't be initialized yet.
    --i;
    assert(i < 3 && (waitersCompletelyInitialized.load() & (1 << i)) == 0 &&
           "Invalid slot #");
    new (waiters + i) Waiter(std::move(waiter));

    // FIXME: This likely doesn't need to be sequentially consistent.
    waitersCompletelyInitialized.fetch_or(1 << i, std::memory_order_seq_cst);
  }

  Waiter takeFirstWaiter() { return std::move(firstWaiter); }

  Waiter takeAndDestroyWaiterN(size_t i) {
    assert(i != 0 && i < 4);
    --i;
    Waiter result = std::move(waiters[i]);
    waiters[i].~Waiter();
    return result;
  }

  void spinUntilWaitersAreInitialized(size_t numWaiters) {
    // Entry 0 is special, it is always valid.  Check the rest of the three
    // using the bitmask in `waitersCompletelyInitialized`.
    if (numWaiters == 1)
      return;
    --numWaiters;
    uint8_t allReadyMask = (1 << numWaiters) - 1;
    // Make sure the waiter in question finished construction.
    SpinWaiter<> spinWaiter;
    while (waitersCompletelyInitialized.load() != allReadyMask) {
      // If not, wait a bit and retry.
      spinWaiter.wait();
    }
  }

private:
  // This waiter is initialized on construction and not tracked in
  // `waitersCompletelyInitialized`.
  Waiter firstWaiter;
  union {
    // These are not implicitly constructed.
    AsyncValue::Waiter waiters[3];
  };

  WaiterListNode *next = nullptr;

  /// This contains a bitset that indicates which elements of 'waiters' have
  /// finished initializing.  This is used when tearing down the list to avoid
  /// races between adding waiters and a value becoming ready.
  /// NOTE: We only use 3 bits in this, we could mash it into the 'next' field
  /// with PointerIntPair if there is a reason to.
  std::atomic<uint8_t> waitersCompletelyInitialized;

  WaiterListNode(const WaiterListNode &) = delete;
  void operator=(const WaiterListNode &) = delete;
};
} // namespace LLCL

/// Invoke all of the waiters specified by the list of waiter nodes, and
/// deallocate the nodes.  We know we have ownership of `list` here, but there
/// may be concurrent mutations.  We cannot know the waiters are settled unless
/// their corresponding `waitersCompletelyInitialized` bit is set.
///
/// We also know that the caller has an RCRef that keeps 'this' alive.
///
void AsyncValue::runWaitersAndDeallocate(WaiterListNode *list,
                                         size_t numEntriesValid) {
  while (list) {
    auto *node = list;
    // The first waiter in a node is always valid.
    assert(numEntriesValid != 0);
    runOneWaiter(node->takeFirstWaiter());

    // If the waiters in the specified node haven't finished initializing, wait
    // for them.
    node->spinUntilWaitersAreInitialized(numEntriesValid);

    // Run waiters 1-3 if they are valid.
    for (size_t i = 1; i != numEntriesValid; ++i)
      runOneWaiter(node->takeAndDestroyWaiterN(i));
    list = node->next;
    delete node;

    // Beyond the first node, we know that all entries are valid.
    numEntriesValid = 4;
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
  SpinWaiter<> spinWaiter;
  while (!compareExchangeWaiterAndState(oldValue, newValue)) {
    // If the thing changed to a different state underneath us, return
    // failure.
    if (oldValue.getInt() != origState)
      return M::failure();

    // If the waiter list changed out from under us, try again.
    if (spinWaiter.wait()) {
      oldValue = loadWaitersAndState();
      if (oldValue.getInt() != origState)
        return M::failure();
    }
    newValue = WaitersAndState(oldValue.getPointer(), newState);
  }
  oldValue = newValue;
  return M::success();
}

/// Given an AsyncValue::State in one of the
/// `kUnconstructed*AvailableOOLWaiterSlots` states, return the number of
/// waiters that are initialized.
static unsigned getNumWaitersValid(AsyncValue::State state) {
  assert(AsyncValue::hasInlineWaiter(state));
  return (int)state -
         int(AsyncValue::State::kUnconstructed1ValidOOLWaiterSlots) + 1;
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
                            State::kUnconstructedInitializingInlineWaiter))) {
      // In the vastly most common case we get into the 'WaiterConstructing'
      // state. Inline initialize the waiter.
      new (getInlineWaiterPointer()) Waiter(std::move(waiter));
      // Then transition immediately to the 'WaiterPresent' state.  We want this
      // critical section to be extremely short, only a few cycles.  We could
      // use `moveState` here, but we know the old value cannot change while
      // in kUnconstructedConstructingInlineWaiter state, and thus this can
      // never fail.  An atomic add is faster and simpler than compare/xchg and
      // loop overhead.
      static_assert(int(State::kUnconstructedInitializingInlineWaiter) + 4 ==
                    int(State::kUnconstructed4ValidOOLWaiterSlots));
      // FIXME: This probably doesn't have to be std::memory_order_seq_cst.
      auto prevValue = waitersAndState.fetch_add(4, std::memory_order_seq_cst);

      // Verify that no one moved the state or waiter list behind our back.
      assert(prevValue == (intptr_t)oldValue.getOpaqueValue() &&
             "nothing can move the value in the WaiterConstructing state");
      // Verify that adding four does actually move to the right state without
      // changing the waiter list.
      assert(prevValue + 4 == (intptr_t)WaitersAndState(
                                  oldValue.getPointer(),
                                  State::kUnconstructed4ValidOOLWaiterSlots)
                                  .getOpaqueValue() &&
             "adding one should get us to the next state");
      (void)prevValue;
      return;
    }
  }

  SpinWaiter<> spinWaiter;
  while (1) {
    // If we raced with a transition into a ready state then we can just execute
    // the waiter and be done.
    if (isReady(oldValue.getInt()))
      return runOneWaiter(std::move(waiter));

    // If there are slots available in the existing waiter node, take one.
    unsigned numValid = getNumWaitersValid(oldValue.getInt());
    if (numValid < 4) {
      // Try to move the state to say that we're claiming this waiter slot.
      if (failed(moveState(oldValue, (State)(int(oldValue.getInt()) + 1)))) {
        spinWaiter.wait();
        continue; // retry on failure.
      }

      // If we succeeded, set the waiter and we're done.
      oldValue.getPointer()->setWaiter(numValid, std::move(waiter));
      return;
    }

    // Otherwise, put the waiter into a new waiter node which will point to the
    // full list as its tail.
    auto node = new WaiterListNode(std::move(waiter), oldValue.getPointer());

    // Swap the next link in. oldValue.getInt() must be non-ready when
    // evaluating the loop condition. The acquire barrier on the
    // compare_exchange ensures that prior changes to waiter list are visible
    // here as we may call RunWaiter() on it. The release barrier ensures that
    // prior changes to *node appear to happen before it's added to the list.
    auto newValue =
        WaitersAndState(node, State::kUnconstructed1ValidOOLWaiterSlots);

    if (compareExchangeWaiterAndState(oldValue, newValue)) {
      // We successfully installed the new node. The oldValue must be in some
      // non-ready state.
      assert(!isReady(oldValue.getInt()));
      return;
    }

    // While swapping in our waiter, the value could have become ready.  If
    // so, just run the waiter and deallocate the node we don't need anymore.
    if (isReady(oldValue.getInt())) {
      assert(oldValue.getPointer() == nullptr);
      runOneWaiter(node->takeFirstWaiter());
      // Change the tail of the list to null.  Whatever moved this to a ready
      // state will already have executed and deallocated the list tail.
      node->next = nullptr;
      delete node;
      return;
    }

    // Otherwise, someone beat us to it and added a new node.  Deallocate our
    // node and try adding the waiter to their node.
    waiter = node->takeFirstWaiter();
    // Change the tail of the list to null.  Whatever moved this to a ready
    // state will already have executed and deallocated the list tail.
    node->next = nullptr;
    delete node;
  }
}

/// Transition to a ready state and notify all waiters about this.  This
/// returns the old state.
AsyncValue::State AsyncValue::notifyReady(State newState,
                                          llvm::Optional<Waiter> &extraWaiter) {
  assert((newState == State::kAvailable || newState == State::kError) &&
         "new state isn't a ready state!");

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto oldValue = exchangeWaiterAndState(WaitersAndState(nullptr, newState));

  // If there was an inline waiter, run it first.
  if (extraWaiter.has_value())
    runOneWaiter(std::move(*extraWaiter));

  // Figure out how many waiters are valid in the first node of the list.
  size_t numEntriesValid = getNumWaitersValid(oldValue.getInt());

  //  Then run the rest of the waiter list.
  runWaitersAndDeallocate(oldValue.getPointer(), numEntriesValid);
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

  // This must have been in one of the unconstructed states, but couldn't have
  // been in kUnconstructed because that would allow a race for another inline
  // waiter to be added. `removeAnyInlineWaiter` ensures this isn't possible.
  assert(hasInlineWaiter(oldState) &&
         "AsyncValue transitioned to while we're changing to error?");
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
