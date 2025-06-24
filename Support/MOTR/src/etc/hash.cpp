//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <string>

#include "Hash.h"

int main() {
  // Use the TOKEN_HASH macro with a sample string
  uint64_t h[8];
  h[0] = MOTR_TOKEN_HASH("example_string"),
  h[1] = MOTR_TOKEN_HASH("example_string"),
  h[2] = MOTR_TOKEN_HASH("example_string3"),
  h[3] = MOTR_TOKEN_HASH("example_string3");
  h[4] = MOTR_TOKEN_HASH("example_string"),
  h[5] = MOTR_TOKEN_HASH("example_string"),
  h[6] = MOTR_TOKEN_HASH("example_string3"),
  h[7] = MOTR_TOKEN_HASH("example_string4");
  constexpr auto lorem = MOTR_TOKEN_HASH("Lorem ipsum dolor sit amet.");
  constexpr auto withnew = MOTR_TOKEN_HASH("Test \
    next line");

  MOTR_TOKEN_HASH("test");
  MOTR_TOKEN_HASH("test");
  MOTR_TOKEN_HASH("testing ");
  MOTR_TOKEN_HASH("test");

  for (auto i = 0; i < 8; i++) {
    printf("Hash%d: 0x%016llX\n", i,
           h[i]); // Output the hash in hexadecimal format
  }
  printf("Lorem: 0x%016llX\n", lorem); // Output the hash in hexadecimal format
  printf("Withnew: 0x%016llX\n",
         withnew); // Output the hash in hexadecimal format
  printf("%s\n", "Test \
    next line");
  return 0;
}
