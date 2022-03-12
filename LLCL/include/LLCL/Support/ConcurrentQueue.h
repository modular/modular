//===- ConcurrentQueue.h --------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_SUPPORT_CONCURRENTQUEUE_H
#define LLCL_SUPPORT_CONCURRENTQUEUE_H

#include <mutex>

namespace LLCL {
/// This class provides a concurrent-safe ordered queue. Items in the queue
/// cannot be re-ordered.
template <typename T>
class ConcurrentQueue {
public:
  ConcurrentQueue() : head(nullptr), tail(nullptr) {}
  ~ConcurrentQueue() { assert(empty() && "Cannot destroy a non-empty queue!"); }

  /// Enqueue takes ownership of the object and ties its lifetime to the
  /// lifetime of the queue.
  void enqueue(T &&obj);

  /// Dequeue returns failure if the queue is empty, otherwise returns the
  /// stored value to the caller and relinquishes ownership.
  mlir::FailureOr<T> dequeue();

  /// This function checks if the queue is empty. It holds the lock, so be
  /// careful where you call it from.
  bool empty() {
    std::lock_guard lock(m);
    return emptyImpl();
  }

private:
  ConcurrentQueue(const ConcurrentQueue &other) = delete;
  ConcurrentQueue &operator=(const ConcurrentQueue &other) = delete;

  /// Linked list node used for this queue.
  struct Item {
    Item *next;
    T data;
  };

  /// The caller must hold the lock for this function, which is why it's
  /// private.
  bool emptyImpl() const { return head == nullptr && head == tail; }

  std::mutex m;
  Item *head;
  Item *tail;
};

//===----------------------------------------------------------------------===//
// Queue function implementations
//===----------------------------------------------------------------------===//

template <typename T>
void ConcurrentQueue<T>::enqueue(T &&obj) {
  auto *newItem = new Item{.next = nullptr, .data = std::move(obj)};
  assert(newItem != nullptr);

  std::lock_guard lock(m);
  Item *prevTail = tail;
  // If the queue is empty then set head and tail to the new item.
  if (emptyImpl())
    head = newItem;
  else
    prevTail->next = newItem;

  tail = newItem;
}

template <typename T>
mlir::FailureOr<T> ConcurrentQueue<T>::dequeue() {
  std::lock_guard lock(m);
  if (emptyImpl())
    return mlir::failure();

  Item *out = head;
  head = out->next;
  // If head has caught up to tail, then set tail to nullptr as well.
  if (head == nullptr)
    tail = nullptr;

  T outData = std::move(out->data);
  delete out;
  return outData;
}

} // namespace LLCL

#endif // LLCL_SUPPORT_CONCURRENTQUEUE_H
