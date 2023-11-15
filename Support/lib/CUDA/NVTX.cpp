//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/CUDA/NVTX.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/DynamicLibrary.h"

using namespace M;
using namespace M::CUDA::NVTX;

static constexpr int kNVTXVersion = 3;

namespace {

/// Both EventPayload and EventAttributes match the layout of
/// https://nvidia.github.io/NVTX/doxygen/group___m_a_r_k_e_r_s___a_n_d___r_a_n_g_e_s.html#gaa31a1079a237d0772f84c56aeda7b26d
/// and/or their definition in the nvToolsExt.h file.
using EventPayload = union {
  uint64_t ullValue;
  int64_t llValue;
  double dValue;
  /* NVTX_VERSION_2 */
  uint32_t uiValue;
  int32_t iValue;
  float fValue;
};

struct EventAttributes {
  /// Version flag of the structure.
  uint16_t version = kNVTXVersion;

  /// Size of the structure.
  uint16_t size = sizeof(EventAttributes);

  /// ID of the category the event is assigned to.
  uint32_t category = 0;

  /// Color type specified in this attribute structure.
  int32_t colorType = 1; // ARGB

  /// Color assigned to this event.
  uint32_t color = 0xFF880000;

  /// Payload type specified in this attribute structure.
  int32_t payloadType = 0;

  int32_t reserved0 = 0;

  /// Payload assigned to this event.
  EventPayload payload;

  /// Message type specified in this attribute structure.
  int32_t messageType = 1; // Ascii.

  /// Message assigned to this attribute structure.
  const char *asciiMessage = nullptr;
};
}; // namespace

#ifdef USE_NVTX_LIB
static ErrorOr<llvm::sys::DynamicLibrary> getNVMLLibraryHandle() {
  std::string errorMessage;
  static auto library = llvm::sys::DynamicLibrary::getPermanentLibrary(
      NVTX_LIBRARY_PATH, &errorMessage);

  if (library.isValid())
    return library;
  return Error(Twine("failed to load NVML library: ") + errorMessage);
}

template <typename SymbolTy>
M::ErrorOr<SymbolTy> fallibleGetSymbol(std::string_view symbolName) {
  static_assert(std::is_pointer_v<SymbolTy>, "Should be a pointer type");

  ErrorOr<llvm::sys::DynamicLibrary> nvmlLib = getNVMLLibraryHandle();
  if (nvmlLib.isError())
    return nvmlLib.takeError();

  void *symbolOr = nvmlLib->getAddressOfSymbol(symbolName.data());
  if (!symbolOr)
    return M::Error(Twine("failed to get symbol '") + symbolName +
                    "' from the NVTX library");
  return reinterpret_cast<SymbolTy>(symbolOr);
}
#endif // USE_NVTX_LIB

ErrorOrSuccess Event::mark() {
#ifdef USE_NVTX_LIB
  static auto nvtxMarkEx =
      fallibleGetSymbol<void (*)(EventAttributes *)>("nvtxMarkEx");
  if (!nvtxMarkEx)
    return Error("unable to load NVML library symbol 'nvtxMarkEx'");

  // Set the marker attributes.
  EventAttributes attr;
  attr.payload.iValue = 0;
  attr.color = color;
  attr.asciiMessage = message.c_str();

  (*nvtxMarkEx)(&attr);
#endif // USE_NVTX_LIB

  return success();
}

ErrorOr<Event::RangeID> Event::start() {
#ifndef USE_NVTX_LIB
  static auto nvtxRangeStartEx =
      fallibleGetSymbol<uint64_t (*)(EventAttributes *)>("nvtxRangeStartEx");
  if (!nvtxRangeStartEx)
    return Error("unable to load NVML library symbol 'nvtxRangeStartEx'");

  // Set the range attributes.
  EventAttributes attr;
  attr.payload.iValue = 0;
  attr.color = color;
  attr.asciiMessage = message.c_str();

  return Event::RangeID((*nvtxRangeStartEx)(&attr));
#else  // USE_NVTX_LIB
  return Event::RangeID(-1);
#endif // USE_NVTX_LIB
}

Event::RangeID::~RangeID() {
#ifdef USE_NVTX_LIB
  static auto nvtxRangeEnd =
      fallibleGetSymbol<void (*)(uint64_t)>("nvtxRangeEnd");
  assert(nvtxRangeEnd && "unable to load NVML library symbol 'nvtxRangeEnd'");
  (*nvtxRangeEnd)(id);
#endif
}
