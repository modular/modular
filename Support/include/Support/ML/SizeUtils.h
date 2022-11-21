//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Helpers for working with size, alignment and dimension values at both
// compile time and runtime using various encoding strategies.
//
// At compile time we have three representations:
//  - Optional<uint64_t> (aka 'clean form'), where none denotes 'unknown', and
//    the value cannot be zero. This is most convenient in the compiler.
//  - uint64_t (aka 'raw form'), where ~0 denotes 'unknown', and the value
//    cannot be zero. This is a BEF-friendly representation, since BEF has
//    no support for optional attributes.
//  - int64_t (aka 'raw signed form'), where the MLIR ShapedType::kDynamicSize
//    value denotes 'unknown', and the value must otherwise be strictly
//    positive. This matches the MLIR ShapedType convention, except we consider
//    0 illegal.
//
// At run time we have the representation:
//  - size_t (aka 'runtime form'), where ~0 denotes 'unknown'.
// (However note the MLIR index type is represented as ssize_t at runtime.)
//
// The utilities here help validate and translate these encodings.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_ML_SIZEUTILS_H
#define SUPPORT_ML_SIZEUTILS_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/Support/Alignment.h"

#include <cstdint>
#include <limits>

namespace M {

/// Denotes an unknown size in the 'raw form' encoding.
constexpr uint64_t kUnknownSize = ~0;

/// Denotes an unknown size in the 'raw signed form' encoding.
/// Copied from ShapedType::kDynamicSize so as to avoid dependency.
constexpr int64_t kUnknownSignedSize = std::numeric_limits<int64_t>::min();

/// Denotes an unknown size in the 'runtime form' encoding.
constexpr size_t kRuntimeUnknownSize = ~0;

/// Returns true if size in clean form is specified.
inline bool hasSize(Optional<uint64_t> size) { return size.has_value(); }

/// Returns true if size in clean from is valid.
inline bool isValidSize(Optional<uint64_t> size) { return !size || *size > 0; }

/// Returns true if size in raw form is specified.
inline bool hasRawSize(uint64_t rawSize) { return rawSize != kUnknownSize; }

/// Returns true if size in raw form is valid.
inline bool isValidRawSize(uint64_t rawSize) {
  return rawSize == kUnknownSize || rawSize > 0;
}

/// Returns true if size in raw signed form is specified.
inline bool hasRawSignedSize(int64_t rawSignedSize) {
  return rawSignedSize != kUnknownSignedSize;
}

/// Returns true if size in raw signed form is valid.
inline bool isValidRawSignedSize(int64_t rawSignedSize) {
  return rawSignedSize == kUnknownSignedSize || rawSignedSize > 0;
}

/// Translates size from clean form to raw form.
/// We assert check for validity, so there's no checking in release builds.
inline uint64_t asRawSize(Optional<uint64_t> size) {
  assert(isValidSize(size) && "invalid size");
  if (!size)
    return kUnknownSize;
  assert(*size != kUnknownSize && "cannot represent size as raw size");
  return *size;
}

/// Translates size from clean form to raw signed form.
/// We assert check for validity, so there's no checking in release builds.
inline int64_t asRawSignedSize(Optional<uint64_t> size) {
  assert(isValidSize(size) && "invalid size");
  if (!size)
    return kUnknownSignedSize;
  assert(*size != kUnknownSize && "cannot represent size as raw size");
  assert(*size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
         "size is too large to represent in signed form");
  return static_cast<int64_t>(*size);
}

/// Translates size from raw form to clean form.
/// We assert check for validity, so there's no checking in release builds.
inline Optional<uint64_t> asCleanSize(uint64_t rawSize) {
  assert(isValidRawSize(rawSize) && "invalid raw form size");
  return rawSize == kUnknownSize ? Optional<uint64_t>() : rawSize;
}

/// Translates size from raw signed form to clean form.
/// We assert check for validity, so there's no checking in release builds.
inline Optional<uint64_t> asCleanSize(int64_t rawSignedSize) {
  assert(isValidRawSignedSize(rawSignedSize) && "invalid raw form size");
  return rawSignedSize == kUnknownSignedSize
             ? Optional<uint64_t>()
             : static_cast<uint64_t>(rawSignedSize);
}

/// Translates size from raw form to runtime form.
/// We assert check for validity and no overflow, so there's no checking in
/// release builds.
inline size_t asRuntimeSize(uint64_t rawSize) {
  assert(isValidSize(rawSize) && "invalid raw form size");
  if (rawSize == kUnknownSize)
    return kRuntimeUnknownSize;
  assert(rawSize <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) &&
         "raw size is too large to represent in runtime form");
  size_t runtimeSize = static_cast<size_t>(rawSize);
  assert(runtimeSize != kRuntimeUnknownSize &&
         "cannot represent raw size in runtime form");
  return runtimeSize;
}

/// Translates size from raw form to runtime form. If the raw form is the
/// distinguished unknown value the defaultSize is returned.
/// We assert check for validity, so there's no checking in release builds.
inline size_t asRuntimeSizeOrDefault(uint64_t rawSize, size_t defaultSize) {
  assert(isValidSize(rawSize) && "invalid raw form size");
  if (rawSize == kUnknownSize)
    return defaultSize;
  assert(rawSize <= static_cast<uint64_t>(std::numeric_limits<size_t>::max()) &&
         "raw size too large to represent in runtime form");
  size_t runtimeSize = static_cast<size_t>(rawSize);
  assert(runtimeSize != kRuntimeUnknownSize &&
         "cannot represent raw size in runtime form");
  return static_cast<size_t>(rawSize);
}

/// Returns true if align in clean form is valid.
inline bool isValidAlign(Optional<uint64_t> align) {
  return !align || llvm::isPowerOf2_64(*align);
}

/// Returns true if align in raw form is valid.
inline bool isValidRawAlign(uint64_t rawAlign) {
  return rawAlign == kUnknownSize || llvm::isPowerOf2_64(rawAlign);
}

/// Translates align in clean form to it's llvm::MaybeAlign
/// representation. We assert check for validity (inside llvm::Align), so
/// there's no checking in release builds.
inline llvm::MaybeAlign asMaybeAlign(Optional<uint64_t> align) {
  return llvm::MaybeAlign(align.value_or(0));
}

/// Translates align in llvm::MaybeAlign form to it's clean representation.
inline Optional<uint64_t> fromMaybeAlign(llvm::MaybeAlign align) {
  return align ? align->value() : Optional<uint64_t>();
}

} // namespace M

#endif // SUPPORT_ML_SIZEUTILS_H
