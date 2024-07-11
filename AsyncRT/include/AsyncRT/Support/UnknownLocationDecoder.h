//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef ASYNCRT_SUPPORT_UNKNOWNLOCATIONDECODER_H
#define ASYNCRT_SUPPORT_UNKNOWNLOCATIONDECODER_H

#include "AsyncRT/Support/Location.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/ReferenceCounted.h"

namespace M::AsyncRT {

/// This class implements LocationDecoder and returns a filename of `<unknown>`.
/// This is useful for inner infrastructure that doesn't really have a good way
/// of inferring a location that hasn't been passed-in.
class UnknownLocationDecoder final
    : public ReferenceCounted<UnknownLocationDecoder>,
      public LocationDecoder {
public:
  UnknownLocationDecoder() = default;

  static EncodedLocation getEncodedLocation() {
    return {0, RCRef<UnknownLocationDecoder>::create()};
  }

  static EncodedDiagnostic getDiagnostic(Error err);

  /// This decodes nothing - there's nothing encoded.
  DecodedLocation decode(const EncodedLocation &loc) const override {
    return DecodedLocation{"<unknown>"};
  }

  /// Final LocationDecoder hooks.
  void addRef() const override;
  void dropRef() const override;
};

} // namespace M::AsyncRT

#endif // ASYNCRT_SUPPORT_UNKNOWNLOCATIONDECODER_H
