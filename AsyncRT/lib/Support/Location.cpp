//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements Location.h classes.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Support/Location.h"

using namespace M::LLCL;

void LocationDecoder::VtableAnchor() {}

/// Decode the location information in this object into a DecodedLocation.
DecodedLocation EncodedLocation::decode() const {
  return decoder->decode(*this);
}
