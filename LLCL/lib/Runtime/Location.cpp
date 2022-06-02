//===- Location.cpp -------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements Location.h classes.
//
//===----------------------------------------------------------------------===//

#include "LLCL/Runtime/Location.h"
#include "LLCL/Runtime/AsyncValue.h"
using namespace LLCL;

void LocationDecoder::VtableAnchor() {}

/// Create an error AsyncValue at this location with the specified message.
/// For consistency, the error message should start with a lower case letter
/// and not end with a period.
AnyAsyncValueRef EncodedLocation::createErrorValue(CompactRuntimePtr runtime,
                                                   M::Error message) const {
  return AsyncValue::createError(runtime,
                                 EncodedDiagnostic{std::move(message), copy()});
}

/// Decode the location information in this object into a DecodedLocation.
DecodedLocation EncodedLocation::decode() const {
  return decoder->decode(*this);
}