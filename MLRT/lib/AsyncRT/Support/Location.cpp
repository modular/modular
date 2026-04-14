//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements Location.h classes.
//
//===----------------------------------------------------------------------===//

#include "MLRT/AsyncRT/Support/Location.h"

using namespace M::MLRT;

void LocationDecoder::VtableAnchor() {}

/// Decode the location information in this object into a DecodedLocation.
DecodedLocation EncodedLocation::decode() const {
  return decoder->decode(*this);
}
