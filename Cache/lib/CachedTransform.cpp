//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CachedTransform.h"
#include "LLCL/Runtime/Algorithms.h"
#include "llvm/Support/BLAKE3.h"

using namespace M;
using namespace Cache;
using namespace LLCL;

//===----------------------------------------------------------------------===//
// Generic Transformations
//===----------------------------------------------------------------------===//

std::string TransformCacheKey::hashKey(TransformCacheKey::KeyTy key) {
  // This is just a (usually relatively small) string - the hash is just the
  // SHA256 hash of the input.
  std::array<uint8_t, 32> hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)key->getBufferStart(), key->getBufferSize()));
  return {hash.begin(), hash.end()};
}
