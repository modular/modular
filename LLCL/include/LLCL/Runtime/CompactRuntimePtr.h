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

#include <cassert>
#include <cstdint>

#include "LLCL/Runtime/Globals/CompactRuntimeTable.h"

namespace M::LLCL {

class Runtime;

// CompactRuntimePtr implements a compact pointer for a HostContext by storing
// the instance index of the HostContext object. It is intended to be used in
// places where saving the memory space is important, otherwise, HostContext*
// should be used.
class CompactRuntimePtr {
public:
  CompactRuntimePtr() = default;
  CompactRuntimePtr(const CompactRuntimePtr &) = default;
  CompactRuntimePtr &operator=(const CompactRuntimePtr &) = default;

  // Implicitly convert Runtime* to CompactRuntimePtr.
  /*implicit*/ CompactRuntimePtr(Runtime *runtime);
  /*implicit*/ CompactRuntimePtr(Runtime &runtime)
      : CompactRuntimePtr(&runtime) {}

  Runtime *operator->() const { return get(); }
  Runtime &operator*() const { return *get(); }
  Runtime *get() const { return M::LLCL::Globals::getRuntime(index); }

  static intptr_t getSignature() {
    return M::LLCL::Globals::getRuntimeSignature();
  }

  /// Explicitly testing for truth value determines whether this pointer is
  /// "null".
  explicit operator bool() const { return index != kInvalidIndex; }

  /// We implicitly convert to Runtime& since we are used interchangably with
  /// it.
  operator Runtime &() const { return *get(); }

  /// Get an opaque token for the pointer.
  uint8_t getAsOpaqueToken() const { return index; }
  /// Get the pointer from an opaque token.
  static CompactRuntimePtr getFromOpaqueToken(uint8_t token) {
    return CompactRuntimePtr(token);
  }

  static constexpr uint8_t kInvalidIndex = 255;

private:
  friend class Runtime;
  explicit CompactRuntimePtr(uint8_t index) : index{index} {
    assert(index < kInvalidIndex && "Too many Runtime instances created");
  }
  uint8_t index = kInvalidIndex;
};

} // namespace M::LLCL

#endif // LLCL_RUNTIME_COMPACT_RUNTIME_PTR_H
