//===- Runtime.h - Top-level context for LLCL -------------------*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares top level "god object" that organizes LLCL thread pool,
// memory allocator, etc.
//
// This header file is intended to be a low-dependency header that other things
// compose on top of.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_H
#define LLCL_RUNTIME_H

//#include "llvm/ADT/FunctionExtras.h"
//#include "llvm/Support/Compiler.h"
#include <memory>

namespace LLCL {
class Allocator;

/// This represents one instance of the LLCL runtime, which can have multiple
/// threads, a private heap for data, and a way of reporting errors.  This is
/// also the natural unit for cancelation.
///
class Runtime final {
public:
  // TODO: Diagnostics, thread pool.
  Runtime(std::unique_ptr<Allocator> allocator);
  ~Runtime();

  //===--------------------------------------------------------------------===//
  // Memory Management
  //===--------------------------------------------------------------------===//

  /// Get direct access to the low level allocator.
  Allocator *getAllocator() { return allocator.get(); }

  // Allocate the specified number of bytes with the specified alignment.
  void *allocateBytes(size_t size, size_t alignment);

  // Deallocate the specified pointer that had the specified size.
  void deallocateBytes(void *ptr, size_t size);

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

  //===--------------------------------------------------------------------===//
  // Concurrency
  //===--------------------------------------------------------------------===//

  // TODO

  //===--------------------------------------------------------------------===//
  // Error Reporting
  //===--------------------------------------------------------------------===//

  // TODO

  //===--------------------------------------------------------------------===//
  // Cancel the current execution
  //===--------------------------------------------------------------------===//

  // TODO

private:
  Runtime(const Runtime &) = delete;
  void operator=(const Runtime &) = delete;

  std::unique_ptr<Allocator> allocator;
};

} // namespace LLCL

#endif // LLCL_RUNTIME_H
