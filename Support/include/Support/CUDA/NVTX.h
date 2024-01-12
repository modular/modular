//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Defines a subset of the bindings need to report with NVTX markers.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_CUDA_NVTX_H
#define SUPPORT_CUDA_NVTX_H

#include "Support/ErrorOr.h"
#include "llvm/ADT/StringRef.h"

namespace M::CUDA::NVTX {

struct Event {

  Event(StringRef category, StringRef message, uint32_t color = 0xFF880000)
      : category(category), message(message), color(color) {}

  struct RangeID {
#ifdef USE_NVTX_LIB
    RangeID(uint64_t id) : id(id) {}
    ~RangeID();

  private:
    uint64_t id;
#endif // USE_NVTX_LIB
  };

  ErrorOrSuccess mark();

  ErrorOr<RangeID> start();

private:
  // The cateogry for the event. This is used to disambiguate the different
  // event types.
  std::string category;
  // The message attached to the event.
  std::string message;
  // The ARGB color (in Hex) of the event.
  uint32_t color;
};
} // namespace M::CUDA::NVTX

#endif // SUPPORT_CUDA_NVTX_H
