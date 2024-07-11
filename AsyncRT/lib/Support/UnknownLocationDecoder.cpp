//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Support/UnknownLocationDecoder.h"
#include "AsyncRT/Support/Diagnostic.h"

using namespace M;
using namespace LLCL;

EncodedDiagnostic UnknownLocationDecoder::getDiagnostic(Error err) {
  return {std::move(err), UnknownLocationDecoder::getEncodedLocation()};
}

void UnknownLocationDecoder::addRef() const {
  RCRef<ReferenceCounted<UnknownLocationDecoder>>::lowLevelAddRef(
      const_cast<UnknownLocationDecoder *>(this));
}
void UnknownLocationDecoder::dropRef() const {
  RCRef<ReferenceCounted<UnknownLocationDecoder>>::lowLevelDropRef(
      const_cast<UnknownLocationDecoder *>(this));
}
