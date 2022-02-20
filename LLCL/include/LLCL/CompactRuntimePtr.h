//===- CompactRuntimePtr.h - A `Runtime*` encoded in 8 bits -----*- C++ -*-===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef LLCL_COMPACT_RUNTIME_PTR_H
#define LLCL_COMPACT_RUNTIME_PTR_H

#include <cassert>

namespace LLCL {

class Runtime;

// CompactRuntimePtr implements a compact pointer for a HostContext by storing
// the instance index of the HostContext object. It is intended to be used in
// places where saving the memory space is important, otherwise, HostContext*
// should be used.
class CompactRuntimePtr {
public:
  CompactRuntimePtr() = default;
  CompactRuntimePtr(const CompactRuntimePtr &) = default;

  // Implicitly convert Runtime* to CompactRuntimePtr.
  /*implicit*/ CompactRuntimePtr(Runtime *runtime);

  Runtime *operator->() const { return get(); }
  Runtime &operator*() const { return *get(); }
  Runtime *get() const;

  explicit operator bool() const { return index != kInvalidIndex; }

  static constexpr uint8_t kInvalidIndex = 255;

private:
  friend class Runtime;
  explicit CompactRuntimePtr(uint8_t index) : index{index} {
    assert(index < kInvalidIndex && "Too many Runtime instances created");
  }
  const uint8_t index = kInvalidIndex;
};

} // namespace LLCL

#endif // LLCL_COMPACT_RUNTIME_PTR_H
