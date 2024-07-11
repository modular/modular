//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_CONCURRENTQUEUE_H
#define LLCL_SUPPORT_CONCURRENTQUEUE_H

#include <cassert>
#include <memory>
#include <mutex>

namespace M::AsyncRT {

/// This class provides a concurrent-safe ordered queue. Items in the queue
/// cannot be re-ordered.
template <typename T>
class ConcurrentQueue {
public:
  ConcurrentQueue() : head(nullptr), tail(nullptr) {}
  ~ConcurrentQueue() {
    assert(emptyImpl() && "Cannot destroy a non-empty queue!");
  }

  struct ItemType {
    std::unique_ptr<ItemType> next;
    T data;

    ItemType(ItemType *next, T &&data) : next(next), data(std::move(data)) {}
  };

  /// Enqueue takes ownership of the object and ties its lifetime to the
  /// lifetime of the queue.
  void enqueue(T &&data) {
    std::unique_ptr<ItemType> newItem(new ItemType(nullptr, std::move(data)));
    std::lock_guard<std::mutex> lock(m);

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
  T dequeue() {
    std::lock_guard<std::mutex> lock(m);
    if (emptyImpl())
      return nullptr;

    auto out = std::move(head);
    head = std::move(out->next);

    // If head has caught up to tail, then set tail to nullptr as well.
    if (head == nullptr)
      tail = nullptr;

    return std::move(out->data);
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

} // namespace M::AsyncRT

#endif // LLCL_SUPPORT_CONCURRENTQUEUE_H
