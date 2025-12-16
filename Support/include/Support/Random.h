//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_RANDOM_H
#define SUPPORT_RANDOM_H

#include "Support/ForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include <cstdint>

namespace M {
/// Generate `numBytes` cryptographically-secure random numbers use them to
/// completely fill `buf`. Returns an error if we were unable to generate random
/// numbers for whatever reason.
struct SecureRandomBytesGenerator {
  /// Non-trivial constructors and destructors for Windows.
  SecureRandomBytesGenerator();
  ~SecureRandomBytesGenerator();

  /// Non-copyable, but move-able.
  SecureRandomBytesGenerator(const SecureRandomBytesGenerator &other) = delete;
  SecureRandomBytesGenerator(SecureRandomBytesGenerator &&other) = default;

  /// Actually perform the random number generation.
  ErrorOrSuccess getRandomBytes(MutableArrayRef<uint8_t> buf);

  /// Needed for Windows. On Windows, this is an HCRYPTPROV, which is a typedef
  /// for a pointer.
  void *ctx = nullptr;
};
} // namespace M

#endif // SUPPORT_RANDOM_H
