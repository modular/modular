//===- Allocator.h - Allocator Abstraction ----------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LLCL::Allocator interface, which allows clients of
// LLCL to implement custom allocation and other fancy policies.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_ALLOCATOR_H
#define LLCL_ALLOCATOR_H

#include <memory>

namespace LLCL {
class LLCLAllocator;

class Allocator {
public:
  virtual ~Allocator() {}

  // Allocate the specified number of bytes with the specified alignment.
  virtual void *allocateBytes(size_t size, size_t alignment) = 0;

  // Deallocate the specified pointer that had the specified size.
  virtual void deallocateBytes(void *ptr, size_t size) = 0;

  // Allocate memory for one or more entries of type T.
  template <typename T>
  T *allocate(size_t numElements = 1) {
    return static_cast<T *>(allocateBytes(sizeof(T) * numElements, alignof(T)));
  }

  // Deallocate the memory for one or more entries of type T.
  template <typename T>
  void deallocate(T *ptr, size_t numElements) {
    DeallocateBytes(ptr, sizeof(T) * numElements);
  }

  // Allocate and initialize an object of type T.
  template <typename T, typename... Args>
  T *construct(Args &&...args) {
    T *buf = allocate<T>();
    return new (buf) T(std::forward<Args>(args)...);
  }

  // Destruct and deallocate space for an object of type T.
  template <typename T>
  void destroy(T *t) {
    t->~T();
    deallocate(t);
  }

protected:
  Allocator() = default;
  Allocator(const Allocator &) = delete;
  void operator=(const Allocator &) = delete;

private:
  virtual void vtableAnchor();
};

// Create an allocator that just calls malloc/free.
std::unique_ptr<Allocator> createMallocAllocator();

} // namespace LLCL

#endif // LLCL_ALLOCATOR_H
