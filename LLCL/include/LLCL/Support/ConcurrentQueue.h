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

/// This class provides a concurrent-safe ordered queue. Items in the queue
/// cannot be re-ordered.
class ConcurrentQueue {
public:
  ConcurrentQueue() : head(nullptr), tail(nullptr) {}
  ~ConcurrentQueue() {
    assert(emptyImpl() && "Cannot destroy a non-empty queue!");
  }

  /// Base class for representing an item in the queue.
  /// Users should specifically allocate an instance of Item with a given
  /// anonymous lambda function.
  class ItemBase {
  public:
    explicit ItemBase() : next(nullptr) {}
    virtual ~ItemBase() {}
    virtual void call() = 0;

    std::unique_ptr<ItemBase> next;

  private:
    ItemBase(const ItemBase &) = delete;
    void operator=(const ItemBase &) = delete;
  };

  /// Templated ItemBase implementation class that holds a anonymous lambda
  /// function.
  template <typename CallableT>
  class Item : public ItemBase {
  public:
    explicit Item(CallableT &&newCallable)
        : ItemBase(), callable(std::move(newCallable)) {}

    void call() override { callable(); }

    CallableT callable;
  };

  /// Enqueue takes ownership of the object and ties its lifetime to the
  /// lifetime of the queue.
  void enqueue(ItemBase *item) {
    assert(item != nullptr);
    std::unique_ptr<ItemBase> newItem(item);

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
  std::unique_ptr<ItemBase> dequeue() {
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
  std::unique_ptr<ItemBase> head;
  ItemBase *tail;
};

} // namespace LLCL

#endif // LLCL_SUPPORT_CONCURRENTQUEUE_H
