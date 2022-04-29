//===- ConcurrentQueue.h --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_CONCURRENTQUEUE_H
#define LLCL_SUPPORT_CONCURRENTQUEUE_H

#include <cassert>
#include <memory>
#include <mutex>

namespace LLCL {

/// This class is a default item implementation for ConcurrentQueue, but clients
/// of ConcurrentQueue may provide their own as well.  Implementations must
/// provide a call method and have a next pointer.
class WorkQueueItem {
public:
  explicit WorkQueueItem() : next(nullptr) {}
  virtual ~WorkQueueItem() {}
  virtual void call() = 0;

  std::unique_ptr<WorkQueueItem> next;

private:
  WorkQueueItem(const WorkQueueItem &) = delete;
  void operator=(const WorkQueueItem &) = delete;
};

/// This class provides a concurrent-safe ordered queue. Items in the queue
/// cannot be re-ordered.
template <typename ItemType = WorkQueueItem>
class ConcurrentQueue {
public:
  ConcurrentQueue() : head(nullptr), tail(nullptr) {}
  ~ConcurrentQueue() {
    assert(emptyImpl() && "Cannot destroy a non-empty queue!");
  }

  /// Enqueue takes ownership of the object and ties its lifetime to the
  /// lifetime of the queue.
  void enqueue(ItemType *item) {
    assert(item != nullptr);
    std::unique_ptr<ItemType> newItem(item);

    std::lock_guard lock(m);

    /// If the queue is empty then set head and tail to the new item.
    if (emptyImpl()) {
      head = std::move(newItem);
      tail = head.get();
    } else {
      tail->next = std::move(newItem);
      tail = tail->next.get();
    }
  }

  /// Dequeue returns failure if the queue is empty, otherwise returns the
  /// stored value to the caller and relinquishes ownership.
  std::unique_ptr<ItemType> dequeue() {
    std::lock_guard lock(m);
    if (emptyImpl())
      return nullptr;

    auto out = std::move(head);
    head = std::move(out->next);

    // If head has caught up to tail, then set tail to nullptr as well.
    if (head == nullptr)
      tail = nullptr;

    return out;
  }

private:
  ConcurrentQueue(const ConcurrentQueue &other) = delete;
  ConcurrentQueue &operator=(const ConcurrentQueue &other) = delete;

  /// The caller must hold the lock for this function, which is why it's
  /// private.
  bool emptyImpl() const { return head == nullptr && head.get() == tail; }

  std::mutex m;
  std::unique_ptr<ItemType> head;
  ItemType *tail;
};

} // namespace LLCL

#endif // LLCL_SUPPORT_CONCURRENTQUEUE_H
