//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// A `Runtime*` encoded in 8 bits.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_RUNTIME_COMPACT_RUNTIME_PTR_H
#define LLCL_RUNTIME_COMPACT_RUNTIME_PTR_H

#include "AsyncRT/Runtime/Globals/Globals.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <mutex>

namespace M::AsyncRT {

class Runtime;

namespace Detail {

//===----------------------------------------------------------------------===//
// RuntimeTable
//===----------------------------------------------------------------------===//

/// Global singleton which maintains the runtime index to runtime map.
class RuntimeTable {
public:
  /// Returns runtime with given index, which must have already been added or
  /// registered.
  Runtime *getRuntime(uint8_t index) const;

  /// Reserves an index for a runtime, returning the index. The actual runtime
  /// must be set by setRuntime() below once known.
  uint8_t reserveIndex();

  /// Sets the runtime for the already reserved index.
  void setRuntime(uint8_t index, Runtime *runtime);

  /// Unregistered the runtime with the given index.
  void clearRuntime(uint8_t index);

  /// Returns the number of active runtimes.
  size_t numActiveRuntimes() const;

  /// Index representing 'no runtime'.
  static constexpr uint8_t kInvalidIndex = 255;

  static RuntimeTable &getSingleton() {
    return Globals::getRuntimeTableSingleton(
        []() { return new RuntimeTable(); });
  }

private:
  RuntimeTable();

  /// Protects mutation to both of the following fields.
  mutable std::mutex mu;
  llvm::SmallVector<uint8_t, 256> freeIndices;
  Runtime *allRuntimes[kInvalidIndex];
};

} // namespace Detail

//===----------------------------------------------------------------------===//
// CompactRuntimePtr
//===----------------------------------------------------------------------===//

/// The `CompactRuntimePtr` type provides a pointer compressed version of
/// `Runtime*` that fits in 8 bits.  This allows every AsyncValue to carry a
/// backpointer to the Runtime which allocated it, and allows deallocating the
/// memory for the AsyncValue through the Runtime's allocator.
class CompactRuntimePtr {
public:
  constexpr CompactRuntimePtr() = default;
  CompactRuntimePtr(const CompactRuntimePtr &) = default;
  CompactRuntimePtr &operator=(const CompactRuntimePtr &) = default;

  static CompactRuntimePtr reserve() {
    return CompactRuntimePtr(
        Detail::RuntimeTable::getSingleton().reserveIndex());
  }

  // Implicitly convert Runtime* to CompactRuntimePtr.
  /*implicit*/ CompactRuntimePtr(Runtime *runtime);
  /*implicit*/ CompactRuntimePtr(Runtime &runtime)
      : CompactRuntimePtr(&runtime) {}

  Runtime *operator->() const { return get(); }
  Runtime &operator*() const { return *get(); }
  Runtime *get() const {
    return Detail::RuntimeTable::getSingleton().getRuntime(index);
  }

  Runtime *getOrNull() const {
    return index == Detail::RuntimeTable::kInvalidIndex
               ? nullptr
               : Detail::RuntimeTable::getSingleton().getRuntime(index);
  }

  /// Explicitly testing for truth value determines whether this pointer is
  /// "null".
  explicit operator bool() const {
    return index != Detail::RuntimeTable::kInvalidIndex;
  }

  bool operator==(CompactRuntimePtr that) const { return index == that.index; }

  /// We implicitly convert to Runtime& since we are used interchangably with
  /// it.
  /*implicit*/ operator Runtime &() const { return *get(); }

  /// Returns a 'signature' for the CompactRuntimePtr subsystem which is
  /// expected to be unique for the running process. This can be used to catch,
  /// at runtime, accidental multiple definitions for Modular runtime statics
  /// across dynamic libraries / executables.
  ///
  /// (This is just the address of the underlying runtime table, but
  /// please don't depend on that.)
  static intptr_t getSignature() {
    return reinterpret_cast<intptr_t>(&Detail::RuntimeTable::getSingleton());
  }

  /// Returns the CompactRuntimePtr to the Runtime which is managing the
  /// caller's thread. Returns the invalid CompactRuntimePtr if no such
  /// runtime has been associated.
  static CompactRuntimePtr getCurrentRuntime() {
    return Globals::getCurrentRuntimeInTLS();
  }

  /// Associates the given CompactRuntimePtr with the current thread,
  /// silently overwriting any existing association.
  static void setCurrentRuntime(CompactRuntimePtr ptr) {
    Globals::getCurrentRuntimeInTLS() = ptr;
  }

private:
  friend class Runtime;

  explicit CompactRuntimePtr(uint8_t index) : index{index} {
    assert(index < Detail::RuntimeTable::kInvalidIndex &&
           "Too many Runtime instances created");
  }
  uint8_t index = Detail::RuntimeTable::kInvalidIndex;
};

} // namespace M::AsyncRT

#endif // LLCL_RUNTIME_COMPACT_RUNTIME_PTR_H
