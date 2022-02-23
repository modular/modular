//===- AsyncValue.cpp - Implementation for AsyncValue classes -------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/AsyncValue.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Support/ConcurrentAppendingVector.h"

using namespace LLCL;

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

uint16_t Detail::SomeConcreteAsyncValue::createTypeInfoAndReturnTypeIDImpl(
    DestructorFn destructor) {
  size_t typeID = getTypeInfoTableSingleton().emplace_back(destructor) + 1;
  // Detect overflow.
  assert(typeID < std::numeric_limits<uint16_t>::max() &&
         "too many different AsyncValue types.");
  return typeID;
}

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

void AsyncValue::destroyWithRefCountZero() {
  if (getSubclassKind() == SubclassKind::kIndirect) {
    assert(0 && "indirect async values not implemented yet");
    // delete static_cast<IndirectAsyncValue *>(this);
    return;
  }

  auto *concrete = static_cast<Detail::SomeConcreteAsyncValue *>(this);
  concrete->getDestructor()(concrete);
  M::alignedFree(this);
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
      : waiter(std::move(waiter)), next(nullptr) {}

  llvm::unique_function<void()> waiter;
  // This is the next thing waiting on the AsyncValue.
  WaiterListNode *next;
};
} // namespace LLCL

/// Invoke all of the waiters specified by the list of waiter nodes, and
/// deallocate the waiter nodes.
static void runWaitersAndDeallocate(WaiterListNode *list) {
  while (list) {
    auto *node = list;
    node->waiter();
    list = node->next;
    delete node;
  }
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
  auto oldState = oldValue.getInt();

  // Swap the next link in. oldValue.getInt() must be non-ready when
  // evaluating the loop condition. The acquire barrier on the compare_exchange
  // ensures that prior changes to waiter list are visible here as we may call
  // RunWaiter() on it. The release barrier ensures that prior changes to *node
  // appear to happen before it's added to the list.
  node->next = oldValue.getPointer();
  auto newValue = WaitersAndState(node, oldState);
  while (!waitersAndState.compare_exchange_weak(oldValue, newValue,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
    // While swapping in our waiter, the value could have become ready.  If
    // so, just run the waiter and deallocate the node we don't need anymore.
    if (isReady(oldValue.getInt())) {
      assert(oldValue.getPointer() == nullptr);
      node->next = nullptr;
      runWaitersAndDeallocate(node);
      return;
    }
    // Update the waiter list in newValue.
    node->next = oldValue.getPointer();
  }

  // compare_exchange_weak succeeds. The oldValue must be in some non-ready
  // state.
  assert(!isReady(oldValue.getInt()));
}

/// Transition to a ready state and notify all waiters about this.  This
/// returns the old state.
AsyncValue::State AsyncValue::notifyReady(State newState) {
  assert(newState == State::kAvailable || newState == State::kError);

  // Mark the value as available, ensuring that new queries for the state see
  // the value that got filled in.
  auto oldValue = waitersAndState.exchange(WaitersAndState(nullptr, newState),
                                           std::memory_order_acq_rel);
  runWaitersAndDeallocate(oldValue.getPointer());
  return oldValue.getInt();
}
